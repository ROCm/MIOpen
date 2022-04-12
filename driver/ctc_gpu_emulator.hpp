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

#define NEGATIVE_CUTOFF_VAL (-1e20)

template <typename T>
T logaddexp_gpu(T* x, T* y)
{
    T a = std::max(*x, *y);
    T b = std::min(*x, *y);
    T c = b - a;

    return c <= T(NEGATIVE_CUTOFF_VAL)
               ? std::max(a, T(NEGATIVE_CUTOFF_VAL))
               : std::max(T(a + log(T(1) + exp(b - a))), T(NEGATIVE_CUTOFF_VAL));
}

template <typename T>
T logsumexp_gpu(T* in_vec, size_t length)
{
    auto sum = in_vec[0];
    for(int i = 1; i < length; i++)
        sum = logaddexp_gpu(&(in_vec[i]), &sum);

    return sum;
}

template <typename Tgpu, typename Tref = Tgpu>
void subvec_logsoftmax_gpu(Tgpu* in, Tref* out, size_t in_offset, size_t out_offset, size_t length)
{
    auto itr_in  = in + in_offset;
    auto itr_out = out + out_offset;
    Tgpu max_val = *itr_in;
    for(int i = 1; i < length; i++)
        max_val = std::max(*(itr_in + i), max_val);

    for(int i = 0; i < length; i++)
        *(itr_out + i) = Tref(*(itr_in + i) - max_val);

    Tref sum = logsumexp_gpu(itr_out, length);
    for(int i = 0; i < length; i++)
        *(itr_out + i) = std::max(*(itr_out + i) - sum, Tref(NEGATIVE_CUTOFF_VAL));
}

template <typename T>
void ctc_alpha_gpu(std::vector<int>& probsDesc,
                   T* probs_logits,
                   const T* label,
                   const int label_length,
                   const int input_length,
                   const int class_sz,
                   const int batch_id,
                   const int label_repeat,
                   T* label_prime,
                   T* alpha,
                   T* loss,
                   int blank_lb = 0)
{
    int probs_stride[]  = {probsDesc[3], probsDesc[4], probsDesc[5]};
    int label_prime_len = 2 * label_length + 1;

    for(int i = 0; i < label_length; i++)
        label_prime[2 * i + 1] = label[i];
    blank_lb = blank_lb < 0 ? 0 : (blank_lb >= class_sz ? class_sz - 1 : blank_lb);
    for(int i = 0; i <= label_length; i++)
        label_prime[2 * i] = blank_lb;

    int aidx0 = (label_length + label_repeat - input_length) < 0 ? 0 : 1;
    int aidx1 = 1;
    for(int i = aidx0; i <= aidx1; i++)
    {
        size_t pidx = batch_id * probs_stride[1] + label_prime[i];
        alpha[i]    = probs_logits[pidx];
    }

    for(int j = 1; j < input_length; j++)
    {
        for(int i = 0; i < label_prime_len; i++)
        {
            int lb_cur      = label_prime[i];
            int lb_pre      = label_prime[i - 2];
            size_t pidx     = j * probs_stride[0] + batch_id * probs_stride[1] + lb_cur;
            size_t aidx_ts  = j * label_prime_len + i;
            size_t aidx_t1s = (j - 1) * label_prime_len + i;

            T alpha_t1s2 = alpha[aidx_t1s - 2];
            T alpha_t1s1 = alpha[aidx_t1s - 1];
            T alpha_t1s  = alpha[aidx_t1s];
            T alpha_ts   = i == 0 ? alpha_t1s : logaddexp_gpu(&alpha_t1s, &alpha_t1s1);
            if(i >= 2)
                if(lb_cur != blank_lb && lb_cur != lb_pre)
                    alpha_ts = logaddexp_gpu(&alpha_ts, &alpha_t1s2);

            alpha_ts += probs_logits[pidx];
            alpha[aidx_ts] = std::max(alpha_ts, T(NEGATIVE_CUTOFF_VAL));
        }
    }

    int alpha_size = input_length * label_prime_len;
    *loss          = -logaddexp_gpu(&(alpha[alpha_size - 1]), &(alpha[alpha_size - 2]));
}

template <typename T>
void ctc_gradient_gpu(std::vector<int>& probsDesc,
                      std::vector<int>& gradientsDesc,
                      const T* label_prime,
                      const int label_length,
                      const int input_length,
                      const int class_sz,
                      const int batch_id,
                      const int label_repeat,
                      T* alpha_log,
                      T* probs_logits,
                      T* gradients_logits,
                      T* beta_loss,
                      int blank_lb            = 0,
                      bool is_softmax_applied = true)
{
    int probs_stride[]  = {probsDesc[3], probsDesc[4], probsDesc[5]};
    int grads_stride[]  = {gradientsDesc[3], gradientsDesc[4], gradientsDesc[5]};
    int label_prime_len = 2 * label_length + 1;
    blank_lb            = blank_lb < 0 ? 0 : (blank_lb >= class_sz ? class_sz - 1 : blank_lb);

    int alpha_len     = input_length * label_prime_len;
    float prob_lx_log = logaddexp_gpu(&(alpha_log[alpha_len - 1]), &(alpha_log[alpha_len - 2]));

    std::vector<T> beta_buff0(label_prime_len, T(NEGATIVE_CUTOFF_VAL));
    std::vector<T> beta_buff1(label_prime_len, T(NEGATIVE_CUTOFF_VAL));

    int aidx0 = 1;
    int aidx1 = label_length + label_repeat == input_length ? 1 : 0;

    std::vector<T> grad_temp(class_sz, T(NEGATIVE_CUTOFF_VAL));
    for(int k = aidx1; k <= aidx0; k++)
    {
        int k1     = label_prime_len - 1 - k;
        int lb_cur = label_prime[k1];

        size_t pidx    = (input_length - 1) * probs_stride[0] + batch_id * probs_stride[1] + lb_cur;
        size_t bidx_ts = (input_length - 1) * label_prime_len + k1;

        beta_buff0[k1] = probs_logits[pidx];

        T alpha_temp = alpha_log[bidx_ts];
        alpha_temp += beta_buff0[k1];
        grad_temp[lb_cur] = logaddexp_gpu(&(grad_temp[lb_cur]), &alpha_temp);
    }
    for(int i = 0; i < class_sz; i++)
    {
        size_t pidx = (input_length - 1) * probs_stride[0] + batch_id * probs_stride[1] + i;
        size_t gidx = (input_length - 1) * grads_stride[0] + batch_id * grads_stride[1] + i;

        T probs_logits_pidx = probs_logits[pidx];

        if(is_softmax_applied)
        {
            grad_temp[i] -= probs_logits_pidx;
            grad_temp[i] -= prob_lx_log;
            grad_temp[i] = std::max(grad_temp[i], T(NEGATIVE_CUTOFF_VAL));

            gradients_logits[gidx] = exp(probs_logits_pidx) - exp(grad_temp[i]);
        }
        else
        {
            grad_temp[i] -= (probs_logits_pidx * 2);
            grad_temp[i] -= prob_lx_log;
            grad_temp[i] = std::max(grad_temp[i], T(NEGATIVE_CUTOFF_VAL));

            gradients_logits[gidx] = -exp(grad_temp[i]);
        }
    }

    for(int j = 1; j < input_length; j++)
    {
        int j1 = input_length - 1 - j;
        std::fill(grad_temp.begin(), grad_temp.end(), T(NEGATIVE_CUTOFF_VAL));

        for(int k = 0; k < label_prime_len; k++)
        {
            int k1     = label_prime_len - 1 - k;
            int lb_cur = label_prime[k1];
            int lb_pre = label_prime[k1 + 2];

            size_t pidx    = j1 * probs_stride[0] + batch_id * probs_stride[1] + lb_cur;
            size_t bidx_ts = j1 * label_prime_len + k1;

            T beta_temp = j % 2 == 0 ? beta_buff1[k1] : beta_buff0[k1];
            if(k1 <= label_prime_len - 2)
                beta_temp = logaddexp_gpu(
                    &beta_temp, j % 2 == 0 ? &(beta_buff1[k1 + 1]) : &(beta_buff0[k1 + 1]));
            if(k1 <= label_prime_len - 3)
                if(lb_cur != blank_lb && lb_cur != lb_pre)
                    beta_temp = logaddexp_gpu(
                        &beta_temp, j % 2 == 0 ? &(beta_buff1[k1 + 2]) : &(beta_buff0[k1 + 2]));

            beta_temp += probs_logits[pidx];
            beta_temp = std::max(beta_temp, T(NEGATIVE_CUTOFF_VAL));
            if(j % 2 == 0)
                beta_buff0[k1] = beta_temp;
            else
                beta_buff1[k1] = beta_temp;

            beta_temp += alpha_log[bidx_ts];
            grad_temp[lb_cur] = logaddexp_gpu(&(grad_temp[lb_cur]), &beta_temp);
        }

        for(int i = 0; i < class_sz; i++)
        {
            size_t pidx = j1 * probs_stride[0] + batch_id * probs_stride[1] + i;
            size_t gidx = j1 * grads_stride[0] + batch_id * grads_stride[1] + i;

            T probs_logits_pidx = probs_logits[pidx];

            if(is_softmax_applied)
            {
                grad_temp[i] -= probs_logits_pidx;
                grad_temp[i] -= prob_lx_log;
                grad_temp[i] = std::max(grad_temp[i], T(NEGATIVE_CUTOFF_VAL));

                gradients_logits[gidx] = exp(probs_logits_pidx) - exp(grad_temp[i]);
            }
            else
            {
                grad_temp[i] -= (probs_logits_pidx * 2);
                grad_temp[i] -= prob_lx_log;
                grad_temp[i] = std::max(grad_temp[i], T(NEGATIVE_CUTOFF_VAL));

                gradients_logits[gidx] = -exp(grad_temp[i]);
            }
        }
    }
    *beta_loss = input_length % 2 == 0 ? logaddexp_gpu(&(beta_buff1[0]), &(beta_buff1[1]))
                                       : logaddexp_gpu(&(beta_buff0[0]), &(beta_buff0[1]));
}

template <typename Tgpu, typename Tref = Tgpu>
void launchCTCLoss(const int class_sz,
                   const int batch_size,
                   const int max_time_step,
                   const int max_label_len,
                   const int total_label_len,
                   std::vector<int>& probsDesc,
                   std::vector<int>& gradientsDesc,
                   std::vector<Tgpu>& probs,
                   std::vector<Tref>& losses_gpu,
                   std::vector<Tref>& gradients_gpu,
                   std::vector<Tref>& workspace_gpu,
                   std::vector<Tref>& beta_loss,
                   const int blank_lb      = 0,
                   bool is_softmax_applied = true)
{
    int max_S_len       = 2 * max_label_len + 1;
    int lb_prime_offset = 4 * batch_size + total_label_len;
    int problog_offset  = lb_prime_offset + batch_size * max_S_len;
    int alpha_offset    = problog_offset + class_sz * batch_size * max_time_step;
    std::fill(workspace_gpu.begin() + alpha_offset,
              workspace_gpu.begin() + alpha_offset + max_time_step * batch_size * max_S_len,
              Tref(NEGATIVE_CUTOFF_VAL));

    if(is_softmax_applied)
        for(int j = 0; j < max_time_step * batch_size; j++)
            subvec_logsoftmax_gpu(&(probs[0]),
                                  &(workspace_gpu[problog_offset]),
                                  j * class_sz,
                                  j * class_sz,
                                  class_sz);
    else
        std::copy(probs.begin(), probs.end(), workspace_gpu.begin() + problog_offset);

    for(int j = 0; j < batch_size; j++)
    {
        int input_len     = workspace_gpu[j];
        int label_len     = workspace_gpu[batch_size + j];
        int label_offsets = workspace_gpu[2 * batch_size + j];
        int label_repeat  = workspace_gpu[3 * batch_size + j];
        auto lab_begin    = &(workspace_gpu[4 * batch_size]) + label_offsets;
        std::vector<int> indiv_lab(lab_begin, lab_begin + label_len);

        int alpha_offset_j    = alpha_offset + j * max_time_step * max_S_len;
        int lb_prime_offset_j = lb_prime_offset + j * max_S_len;
        ctc_alpha_gpu(probsDesc,
                      &(workspace_gpu[problog_offset]),
                      &(workspace_gpu[4 * batch_size + label_offsets]),
                      label_len,
                      input_len,
                      class_sz,
                      j,
                      label_repeat,
                      &(workspace_gpu[lb_prime_offset_j]),
                      &(workspace_gpu[alpha_offset_j]),
                      &(losses_gpu[j]),
                      blank_lb);

        ctc_gradient_gpu(probsDesc,
                         gradientsDesc,
                         &(workspace_gpu[lb_prime_offset_j]),
                         label_len,
                         input_len,
                         class_sz,
                         j,
                         label_repeat,
                         &(workspace_gpu[alpha_offset_j]),
                         &(workspace_gpu[problog_offset]),
                         &(gradients_gpu[0]),
                         &(beta_loss[j]),
                         blank_lb,
                         is_softmax_applied);
    }
}

template <typename Tgpu, typename Tref = Tgpu>
void RunCTCLossGPUEmulator(std::vector<int>& probsDesc,
                           std::vector<Tgpu>& probs,
                           const int* labels,
                           const int* labelLengths,
                           const int* inputLengths,
                           std::vector<Tref>& losses_gpu,
                           std::vector<int>& gradientsDesc,
                           std::vector<Tref>& gradients_gpu,
                           std::vector<Tref>& workspace_gpu,
                           std::vector<Tref>& beta_loss,
                           const int blank_lb      = 0,
                           bool is_softmax_applied = true)
{
    if(probsDesc[0] != gradientsDesc[0] || probsDesc[1] != gradientsDesc[1] ||
       probsDesc[2] != gradientsDesc[2])
    {
        printf("probs tensor's dimension does not gradients tensor's dimension\n");
        return;
    }

    int class_sz      = probsDesc[2];
    int batch_size    = probsDesc[1];
    int max_time_step = probsDesc[0];
    std::vector<int> repeat(batch_size, 0);
    std::vector<int> labels_offset(batch_size, 0);
    int max_label_len   = 0;
    int total_label_len = 0;

    for(int i = 0; i < batch_size; i++)
    {
        if(inputLengths[i] > max_time_step)
        {
            printf("Wrong input time step at batch : %d \n", i);
            return;
        }
        max_label_len = std::max(max_label_len, labelLengths[i]);
        total_label_len += labelLengths[i];
        labels_offset[i] = i == 0 ? 0 : (labels_offset[i - 1] + labelLengths[i - 1]);

        for(int j = 0; j < labelLengths[i]; j++)
        {
            if(labels[labels_offset[i] + j] >= class_sz)
            {
                printf("Wrong label id at batch : %d \n", i);
                return;
            }
            if(j > 0)
                if(labels[labels_offset[i] + j] == labels[labels_offset[i] + j - 1])
                    repeat[i]++;
        }

        if(labelLengths[i] + repeat[i] > inputLengths[i])
        {
            printf("Error: label length exceeds input time step at batch : %d \n", i);
            return;
        }
    }

    if(probs.size() != (max_time_step * batch_size * class_sz))
    {
        printf("Wrong probability tensor size\n");
        return;
    }
    if(probs.size() != gradients_gpu.size())
    {
        printf("Wrong gradient tensor size\n");
        return;
    }

    // input length
    std::copy(inputLengths, inputLengths + batch_size, workspace_gpu.begin());

    // label length
    std::copy(labelLengths, labelLengths + batch_size, workspace_gpu.begin() + batch_size);

    // label offset
    std::copy(labels_offset.begin(),
              labels_offset.begin() + batch_size,
              workspace_gpu.begin() + 2 * batch_size);

    // label repeat
    std::copy(repeat.begin(), repeat.begin() + batch_size, workspace_gpu.begin() + 3 * batch_size);

    // labels
    std::copy(labels, labels + total_label_len, workspace_gpu.begin() + 4 * batch_size);

    launchCTCLoss(class_sz,
                  batch_size,
                  max_time_step,
                  max_label_len,
                  total_label_len,
                  probsDesc,
                  gradientsDesc,
                  probs,
                  losses_gpu,
                  gradients_gpu,
                  workspace_gpu,
                  beta_loss,
                  blank_lb,
                  is_softmax_applied);
}
