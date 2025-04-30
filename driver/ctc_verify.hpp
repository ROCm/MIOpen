/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once

#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include <array>
#include "ctc_gpu_emulator.hpp"

#define NEGATIVE_CUTOFF_VAL (-1e20)

template <typename T>
T logaddexp(T x, T y)
{
    T a = std::max(x, y);
    T b = std::min(x, y);
    T c = b - a;

    return c <= T(NEGATIVE_CUTOFF_VAL)
               ? std::max(a, T(NEGATIVE_CUTOFF_VAL))
               : std::max(T(a + log(T(1) + exp(b - a))), T(NEGATIVE_CUTOFF_VAL));
}

template <typename T>
T logsumexp(std::vector<T>& in_vec)
{
    auto sum = in_vec[0];
    for(int i = 1; i < in_vec.size(); i++)
        sum = logaddexp(*(in_vec.begin() + i), sum);

    return sum;
}

template <typename Tgpu, typename Tref = Tgpu>
void subvec_logsoftmax(std::vector<Tgpu>& in,
                       std::vector<Tref>& out,
                       size_t in_offset,
                       size_t out_offset,
                       size_t length)
{
    auto itr_in  = in.begin() + in_offset;
    auto itr_out = out.begin() + out_offset;

    std::vector<Tgpu> sub_in(itr_in, itr_in + length);
    Tgpu max_val = *std::max_element(sub_in.begin(), sub_in.end());
    for(int i = 0; i < length; i++)
        sub_in[i] -= max_val;

    Tgpu sum = logsumexp(sub_in);
    for(int i = 0; i < length; i++)
        *(itr_out + i) = std::max(Tref(sub_in[i] - sum), Tref(NEGATIVE_CUTOFF_VAL));
}

template <typename T>
std::vector<T> ctc_forward_log(std::vector<T>& probs_logits,
                               std::vector<int>& label,
                               const int input_length,
                               const int batch_size,
                               const int class_sz,
                               const int pstr0,
                               const int pstr1,
                               const int pstr2,
                               const int batch_id,
                               int blank_lb = 0)
{
    (void)batch_size;
    int probs_stride[]  = {pstr0, pstr1, pstr2};
    int label_prime_len = 2 * label.size() + 1;
    blank_lb            = blank_lb < 0 ? 0 : (blank_lb >= class_sz ? class_sz - 1 : blank_lb);
    std::vector<int> label_prime(label_prime_len, blank_lb);
    for(int i = 0; i < label.size(); i++)
        label_prime[2 * i + 1] = label[i];

    std::vector<T> alpha(input_length * label_prime_len, T(NEGATIVE_CUTOFF_VAL));

    for(int j = 0; j < input_length; j++)
        for(int i = 0; i < label_prime_len; i++)
        {
            size_t pidx    = j * probs_stride[0] + batch_id * probs_stride[1] + label_prime[i];
            size_t aidx_ts = j * label_prime_len + i;
            if(j == 0 && i > 1)
                break;
            else if(j == 0)
            {
                alpha[aidx_ts] = probs_logits[pidx];
                continue;
            }
            size_t aidx_t1s = (j - 1) * label_prime_len + i;
            alpha[aidx_ts]  = alpha[aidx_t1s];
            if(i >= 1)
                alpha[aidx_ts] = logaddexp(alpha[aidx_ts], alpha[aidx_t1s - 1]);
            if(i >= 2)
                if(label_prime[i] != blank_lb && label_prime[i] != label_prime[i - 2])
                    alpha[aidx_ts] = logaddexp(alpha[aidx_ts], alpha[aidx_t1s - 2]);

            alpha[aidx_ts] += probs_logits[pidx];
            alpha[aidx_ts] = std::max(alpha[aidx_ts], T(NEGATIVE_CUTOFF_VAL));
        }

    return alpha;
}

template <typename T>
std::vector<T> ctc_backward_log(std::vector<T>& probs_logits,
                                std::vector<int>& label,
                                const int input_length,
                                const int batch_size,
                                const int class_sz,
                                const int pstr0,
                                const int pstr1,
                                const int pstr2,
                                const int batch_id,
                                int blank_lb = 0)
{
    (void)batch_size;
    int probs_stride[]  = {pstr0, pstr1, pstr2};
    int label_prime_len = 2 * label.size() + 1;
    blank_lb            = blank_lb < 0 ? 0 : (blank_lb >= class_sz ? class_sz - 1 : blank_lb);
    std::vector<int> label_prime(label_prime_len, blank_lb);
    for(int i = 0; i < label.size(); i++)
        label_prime[2 * i + 1] = label[i];

    std::vector<T> beta(input_length * label_prime_len, T(NEGATIVE_CUTOFF_VAL));

    for(int j = 0; j < input_length; j++)
        for(int i = 0; i < label_prime_len; i++)
        {
            int j1         = input_length - 1 - j;
            int i1         = label_prime_len - 1 - i;
            size_t pidx    = j1 * probs_stride[0] + batch_id * probs_stride[1] + label_prime[i1];
            size_t bidx_ts = j1 * label_prime_len + i1;
            if(j == 0 && i > 1)
                break;
            else if(j == 0)
            {
                beta[bidx_ts] = probs_logits[pidx];
                continue;
            }
            size_t bidx_t1s = (j1 + 1) * label_prime_len + i1;

            beta[bidx_ts] = beta[bidx_t1s];
            if(i1 <= label_prime_len - 2)
                beta[bidx_ts] = logaddexp(beta[bidx_ts], beta[bidx_t1s + 1]);
            if(i1 <= label_prime_len - 3)
                if(label_prime[i1] != blank_lb && label_prime[i1] != label_prime[i1 + 2])
                    beta[bidx_ts] = logaddexp(beta[bidx_ts], beta[bidx_t1s + 2]);

            beta[bidx_ts] += probs_logits[pidx];
            beta[bidx_ts] = std::max(beta[bidx_ts], T(NEGATIVE_CUTOFF_VAL));
        }

    return beta;
}

template <typename T>
void ctc_gradient_log(std::vector<int>& label,
                      const int input_length,
                      const int max_time_step,
                      const int batch_size,
                      const int class_sz,
                      const int pstr0,
                      const int pstr1,
                      const int pstr2,
                      const int gstr0,
                      const int gstr1,
                      const int gstr2,
                      const int batch_id,
                      std::vector<T>& probs_logits,
                      std::vector<T>& gradients_logits,
                      int blank_lb = 0)
{
    int probs_stride[]  = {pstr0, pstr1, pstr2};
    int grads_stride[]  = {gstr0, gstr1, gstr2};
    int label_prime_len = 2 * label.size() + 1;
    blank_lb            = blank_lb < 0 ? 0 : (blank_lb >= class_sz ? class_sz - 1 : blank_lb);
    std::vector<int> label_prime(label_prime_len, blank_lb);
    for(int i = 0; i < label.size(); i++)
        label_prime[2 * i + 1] = label[i];

    std::vector<T> alpha_log = ctc_forward_log(probs_logits,
                                               label,
                                               input_length,
                                               batch_size,
                                               class_sz,
                                               pstr0,
                                               pstr1,
                                               pstr2,
                                               batch_id,
                                               blank_lb);
    std::vector<T> beta_log  = ctc_backward_log(probs_logits,
                                               label,
                                               input_length,
                                               batch_size,
                                               class_sz,
                                               pstr0,
                                               pstr1,
                                               pstr2,
                                               batch_id,
                                               blank_lb);

    float prob_lx_log = logaddexp(alpha_log[alpha_log.size() - 1], alpha_log[alpha_log.size() - 2]);

    for(int j = 0; j < input_length; j++)
        for(int i = 0; i < class_sz; i++)
        {
            size_t pidx            = j * probs_stride[0] + batch_id * probs_stride[1] + i;
            size_t gidx            = j * grads_stride[0] + batch_id * grads_stride[1] + i;
            gradients_logits[gidx] = T(NEGATIVE_CUTOFF_VAL);

            for(int k = 0; k < label_prime_len; k++)
                if(label_prime[k] == i)
                {
                    size_t kidx = j * label_prime_len + k;
                    gradients_logits[gidx] =
                        logaddexp(gradients_logits[gidx], alpha_log[kidx] + beta_log[kidx]);
                }
            gradients_logits[gidx] -= (2 * probs_logits[pidx]);
            gradients_logits[gidx] -= prob_lx_log;
            gradients_logits[gidx] = std::max(gradients_logits[gidx], T(NEGATIVE_CUTOFF_VAL));
            gradients_logits[gidx] = -exp(gradients_logits[gidx]);
        }

    for(int j = input_length; j < max_time_step; j++)
        for(int i = 0; i < class_sz; i++)
        {
            size_t gidx            = j * grads_stride[0] + batch_id * grads_stride[1] + i;
            gradients_logits[gidx] = 0;
        }
}

template <typename T>
void ctc_softmaxlayer_gradient_log(std::vector<int>& label,
                                   const int input_length,
                                   const int batch_size,
                                   const int class_sz,
                                   const int pstr0,
                                   const int pstr1,
                                   const int pstr2,
                                   const int gstr0,
                                   const int gstr1,
                                   const int gstr2,
                                   const int batch_id,
                                   std::vector<T>& probs_logits,
                                   std::vector<T>& gradients_logits,
                                   int blank_lb = 0)
{
    int probs_stride[]  = {pstr0, pstr1, pstr2};
    int grads_stride[]  = {gstr0, gstr1, gstr2};
    int label_prime_len = 2 * label.size() + 1;
    blank_lb            = blank_lb < 0 ? 0 : (blank_lb >= class_sz ? class_sz - 1 : blank_lb);
    std::vector<int> label_prime(label_prime_len, blank_lb);
    for(int i = 0; i < label.size(); i++)
        label_prime[2 * i + 1] = label[i];

    std::vector<T> alpha_log = ctc_forward_log(probs_logits,
                                               label,
                                               input_length,
                                               batch_size,
                                               class_sz,
                                               pstr0,
                                               pstr1,
                                               pstr2,
                                               batch_id,
                                               blank_lb);
    std::vector<T> beta_log  = ctc_backward_log(probs_logits,
                                               label,
                                               input_length,
                                               batch_size,
                                               class_sz,
                                               pstr0,
                                               pstr1,
                                               pstr2,
                                               batch_id,
                                               blank_lb);

    float prob_lx_log = logaddexp(alpha_log[alpha_log.size() - 1], alpha_log[alpha_log.size() - 2]);

    for(int j = 0; j < input_length; j++)
        for(int i = 0; i < class_sz; i++)
        {
            size_t pidx            = j * probs_stride[0] + batch_id * probs_stride[1] + i;
            size_t gidx            = j * grads_stride[0] + batch_id * grads_stride[1] + i;
            gradients_logits[gidx] = T(NEGATIVE_CUTOFF_VAL);

            for(int k = 0; k < label_prime_len; k++)
                if(label_prime[k] == i)
                {
                    size_t kidx = j * label_prime_len + k;
                    gradients_logits[gidx] =
                        logaddexp(gradients_logits[gidx], alpha_log[kidx] + beta_log[kidx]);
                }
            gradients_logits[gidx] -= probs_logits[pidx];
            gradients_logits[gidx] -= prob_lx_log;
            gradients_logits[gidx] = std::max(gradients_logits[gidx], T(NEGATIVE_CUTOFF_VAL));

            gradients_logits[gidx] = exp(probs_logits[pidx]) - exp(gradients_logits[gidx]);
        }
}

template <typename Tgpu, typename Tref = Tgpu>
void RunCTCLossCPUVerify(const int num_class,
                         std::vector<size_t> probsSize,
                         std::vector<size_t> probsStride,
                         std::vector<size_t> gradientsSize,
                         std::vector<size_t> gradientsStride,
                         std::vector<Tgpu>& probs,
                         std::vector<int>& labels,
                         std::vector<int>& labelLengths,
                         std::vector<int>& inputLengths,
                         std::vector<Tref>& losses_host,
                         std::vector<Tref>& gradients_host,
                         std::vector<Tref>& workspace_host,
                         const int blank_lb      = 0,
                         bool is_softmax_applied = true,
                         const int verify_path   = 1)
{
    if(labelLengths.size() != inputLengths.size())
    {
        printf("Label batch size does not match input batch size\n");
        return;
    }

    int class_sz      = num_class + 1;
    int max_time_step = *std::max_element(inputLengths.begin(), inputLengths.end());
    int batch_size    = inputLengths.size();

    if(probs.size() != (max_time_step * batch_size * class_sz))
    {
        printf("Wrong probability tensor size\n");
        return;
    }
    if(probs.size() != gradients_host.size())
    {
        printf("Wrong gradient tensor size\n");
        return;
    }
    if(probsSize[0] != max_time_step || probsSize[1] != batch_size || probsSize[2] != class_sz ||
       gradientsSize[0] != max_time_step || gradientsSize[1] != batch_size ||
       gradientsSize[2] != class_sz)
    {
        printf("Wrong tensor size\n");
        return;
    }

    std::vector<Tref> beta_loss(batch_size, 0);
    if(verify_path == 1)
    {
        std::vector<int> probsDesc     = {max_time_step,
                                      batch_size,
                                      class_sz,
                                      int(probsStride[0]),
                                      int(probsStride[1]),
                                      int(probsStride[2])};
        std::vector<int> gradientsDesc = {max_time_step,
                                          batch_size,
                                          class_sz,
                                          int(gradientsStride[0]),
                                          int(gradientsStride[1]),
                                          int(gradientsStride[2])};

        RunCTCLossGPUEmulator(probsDesc,
                              probs,
                              labels.data(),
                              labelLengths.data(),
                              inputLengths.data(),
                              losses_host,
                              gradientsDesc,
                              gradients_host,
                              workspace_host,
                              beta_loss,
                              blank_lb,
                              is_softmax_applied);
    }
    else
    {
        std::vector<Tref> probs_logits(probs.size(), Tref(NEGATIVE_CUTOFF_VAL));
        std::vector<Tref> gradients_logits(probs.size(), Tref(NEGATIVE_CUTOFF_VAL));
        std::vector<Tref> softmaxlayer_gradients_logit(probs.size(), 0);

        for(int j = 0; j < max_time_step * batch_size; j++)
            subvec_logsoftmax(probs, probs_logits, j * class_sz, j * class_sz, class_sz);

        auto probs_logits_use = is_softmax_applied ? probs_logits : probs;

        std::vector<int> label_offsets(batch_size, 0);
        for(int j = 1; j < batch_size; j++)
            label_offsets[j] = label_offsets[j - 1] + labelLengths[j - 1];

        for(int j = 0; j < batch_size; j++)
        {
            auto lab_begin = labels.begin() + label_offsets[j];
            std::vector<int> indiv_lab(lab_begin, lab_begin + labelLengths[j]);

            std::vector<Tref> alpha_log = ctc_forward_log(probs_logits_use,
                                                          indiv_lab,
                                                          inputLengths[j],
                                                          batch_size,
                                                          class_sz,
                                                          int(probsStride[0]),
                                                          int(probsStride[1]),
                                                          int(probsStride[2]),
                                                          j,
                                                          blank_lb);

            float losses_log =
                -logaddexp(alpha_log[alpha_log.size() - 1], alpha_log[alpha_log.size() - 2]);
            losses_host[j] = losses_log;

            std::vector<Tref> beta_log = ctc_backward_log(probs_logits_use,
                                                          indiv_lab,
                                                          inputLengths[j],
                                                          batch_size,
                                                          class_sz,
                                                          int(probsStride[0]),
                                                          int(probsStride[1]),
                                                          int(probsStride[2]),
                                                          j,
                                                          blank_lb);

            beta_loss[j] = logaddexp(beta_log[0], beta_log[1]);

            if(is_softmax_applied)
                ctc_softmaxlayer_gradient_log(indiv_lab,
                                              inputLengths[j],
                                              batch_size,
                                              class_sz,
                                              int(probsStride[0]),
                                              int(probsStride[1]),
                                              int(probsStride[2]),
                                              int(gradientsStride[0]),
                                              int(gradientsStride[1]),
                                              int(gradientsStride[2]),
                                              j,
                                              probs_logits_use,
                                              softmaxlayer_gradients_logit,
                                              blank_lb);
            else
                ctc_gradient_log(indiv_lab,
                                 inputLengths[j],
                                 max_time_step,
                                 batch_size,
                                 class_sz,
                                 int(probsStride[0]),
                                 int(probsStride[1]),
                                 int(probsStride[2]),
                                 int(gradientsStride[0]),
                                 int(gradientsStride[1]),
                                 int(gradientsStride[2]),
                                 j,
                                 probs_logits_use,
                                 gradients_logits,
                                 blank_lb);
        }

        gradients_host = is_softmax_applied ? softmaxlayer_gradients_logit : gradients_logits;

        (void)probsStride;
        (void)gradientsStride;
        (void)workspace_host;
    }
}

template <typename T>
void GetCTCLossWorkspaceSizeCPU(std::vector<size_t> probsDesc,
                                std::vector<size_t> gradientsDesc,
                                const int* labels,
                                const int* labelLengths,
                                const int* inputLengths,
                                miopenCTCLossAlgo_t ctc_algo,
                                size_t* workSpaceSizeCPU)
{
    (void)ctc_algo;

    if(probsDesc[0] != gradientsDesc[0] || probsDesc[1] != gradientsDesc[1] ||
       probsDesc[2] != gradientsDesc[2])
    {
        printf("Label batch size does not match input batch size\n");
        return;
    }

    int class_sz        = probsDesc[2];
    int batch_size      = probsDesc[1];
    int max_time_step   = probsDesc[0];
    int max_label_len   = 0;
    int total_label_len = 0;
    std::vector<int> repeat(batch_size, 0);
    std::vector<int> labels_offset(batch_size, 0);
    size_t wksp_sz_lb  = 0;
    size_t wksp_sz_dat = 0;

    for(int i = 0; i < batch_size; i++)
    {
        if(inputLengths[i] > max_time_step)
        {
            printf("Wrong input time step\n");
            return;
        }
        max_label_len = std::max(max_label_len, labelLengths[i]);
        total_label_len += labelLengths[i];
        labels_offset[i] = i == 0 ? 0 : (labels_offset[i - 1] + labelLengths[i - 1]);

        for(int j = 0; j < labelLengths[i]; j++)
        {
            if(labels[labels_offset[i] + j] >= class_sz)
            {
                printf("Wrong label id at batch\n");
                return;
            }
            if(j > 0)
                if(labels[labels_offset[i] + j] == labels[labels_offset[i] + j - 1])
                    repeat[i]++;
        }

        if(labelLengths[i] + repeat[i] > inputLengths[i])
        {
            printf("Error: label length exceeds input time step\n");
            return;
        }
    }

    // input length
    wksp_sz_lb += batch_size;

    // label length
    wksp_sz_lb += batch_size;

    // label offset
    wksp_sz_lb += batch_size;

    // label repeat
    wksp_sz_lb += batch_size;

    // labels
    wksp_sz_lb += total_label_len;

    // labels with blanks
    wksp_sz_lb += batch_size * (2 * max_label_len + 1);

    // logsoftmax of probs
    wksp_sz_dat += max_time_step * batch_size * class_sz;

    // alphas
    wksp_sz_dat += max_time_step * batch_size * (2 * max_label_len + 1);

    *workSpaceSizeCPU = (wksp_sz_dat + wksp_sz_lb) * sizeof(T);
}
