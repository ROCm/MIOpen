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

#include "driver.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "rnn_util.hpp"
#include "random.hpp"
#include <array>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <miopen/ctc.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>
#include <cfloat>
#include <algorithm>

#define NEGATIVE_CUTOFF_VAL (-1e20)

template <typename T>
T logaddexp_cpu(T* x, T* y)
{
    T a = std::max(*x, *y);
    T b = std::min(*x, *y);
    T c = b - a;

    return c <= T(NEGATIVE_CUTOFF_VAL)
               ? std::max(a, T(NEGATIVE_CUTOFF_VAL))
               : std::max(T(a + std::log(T(1) + std::exp(b - a))), T(NEGATIVE_CUTOFF_VAL));
}

template <typename T>
T logsumexp_cpu(T* in_vec, size_t length)
{
    auto sum = in_vec[0];
    for(size_t i = 1; i < length; i++)
        sum = logaddexp_cpu(&(in_vec[i]), &sum);

    return sum;
}

template <typename Tgpu, typename Tref = Tgpu>
void subvec_logsoftmax_cpu(Tgpu* in, Tref* out, size_t in_offset, size_t out_offset, size_t length)
{
    auto itr_in  = in + in_offset;
    auto itr_out = out + out_offset;
    Tgpu max_val = *itr_in;
    for(size_t i = 1; i < length; i++)
        max_val = std::max(*(itr_in + i), max_val);

    for(size_t i       = 0; i < length; i++)
        *(itr_out + i) = Tref(*(itr_in + i) - max_val);

    Tref sum = logsumexp_cpu(itr_out, length);
    for(size_t i       = 0; i < length; i++)
        *(itr_out + i) = std::max(*(itr_out + i) - sum, Tref(NEGATIVE_CUTOFF_VAL));
}

template <typename T>
void ctc_alpha_cpu(std::vector<int>& probsDesc,
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

    for(int i                  = 0; i < label_length; i++)
        label_prime[2 * i + 1] = label[i];
    blank_lb = blank_lb < 0 ? 0 : (blank_lb >= class_sz ? class_sz - 1 : blank_lb);
    for(int i              = 0; i <= label_length; i++)
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
            int lb_cur = label_prime[i];
            int lb_pre = -1;
            if(i >= 2)
                lb_pre      = label_prime[i - 2];
            size_t pidx     = j * probs_stride[0] + batch_id * probs_stride[1] + lb_cur;
            size_t aidx_ts  = j * label_prime_len + i;
            size_t aidx_t1s = (j - 1) * label_prime_len + i;

            T alpha_t1s2 = 0;
            if(aidx_t1s >= 2)
                alpha_t1s2 = alpha[aidx_t1s - 2];
            T alpha_t1s1   = 0;
            if(aidx_t1s >= 1)
                alpha_t1s1 = alpha[aidx_t1s - 1];
            T alpha_t1s    = alpha[aidx_t1s];
            T alpha_ts     = i == 0 ? alpha_t1s : logaddexp_cpu(&alpha_t1s, &alpha_t1s1);
            if(i >= 2)
                if(lb_cur != blank_lb && lb_cur != lb_pre)
                    alpha_ts = logaddexp_cpu(&alpha_ts, &alpha_t1s2);

            alpha_ts += probs_logits[pidx];
            alpha[aidx_ts] = std::max(alpha_ts, T(NEGATIVE_CUTOFF_VAL));
        }
    }

    int alpha_size = input_length * label_prime_len;
    *loss          = -logaddexp_cpu(&(alpha[alpha_size - 1]), &(alpha[alpha_size - 2]));
}

template <typename T>
void ctc_gradient_cpu(std::vector<int>& probsDesc,
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
                      int blank_lb            = 0,
                      bool is_softmax_applied = true)
{
    int probs_stride[]  = {probsDesc[3], probsDesc[4], probsDesc[5]};
    int grads_stride[]  = {gradientsDesc[3], gradientsDesc[4], gradientsDesc[5]};
    int label_prime_len = 2 * label_length + 1;
    blank_lb            = blank_lb < 0 ? 0 : (blank_lb >= class_sz ? class_sz - 1 : blank_lb);

    int alpha_len = input_length * label_prime_len;
    assert(alpha_len >= 2);
    float prob_lx_log = logaddexp_cpu(&(alpha_log[alpha_len - 1]), &(alpha_log[alpha_len - 2]));

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
        grad_temp[lb_cur] = logaddexp_cpu(&(grad_temp[lb_cur]), &alpha_temp);
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
                beta_temp = logaddexp_cpu(
                    &beta_temp, j % 2 == 0 ? &(beta_buff1[k1 + 1]) : &(beta_buff0[k1 + 1]));
            if(k1 <= label_prime_len - 3)
                if(lb_cur != blank_lb && lb_cur != lb_pre)
                    beta_temp = logaddexp_cpu(
                        &beta_temp, j % 2 == 0 ? &(beta_buff1[k1 + 2]) : &(beta_buff0[k1 + 2]));

            beta_temp += probs_logits[pidx];
            beta_temp = std::max(beta_temp, T(NEGATIVE_CUTOFF_VAL));
            if(j % 2 == 0)
                beta_buff0[k1] = beta_temp;
            else
                beta_buff1[k1] = beta_temp;

            beta_temp += alpha_log[bidx_ts];
            grad_temp[lb_cur] = logaddexp_cpu(&(grad_temp[lb_cur]), &beta_temp);
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
                   std::vector<Tref>& losses_cpu,
                   std::vector<Tref>& gradients_cpu,
                   std::vector<Tref>& workspace_cpu,
                   const int blank_lb      = 0,
                   bool is_softmax_applied = true)
{
    int max_S_len       = 2 * max_label_len + 1;
    int lb_prime_offset = 4 * batch_size + total_label_len;
    int problog_offset  = lb_prime_offset + batch_size * max_S_len;
    int alpha_offset    = problog_offset + class_sz * batch_size * max_time_step;
    std::fill(workspace_cpu.begin() + alpha_offset,
              workspace_cpu.begin() + alpha_offset + max_time_step * batch_size * max_S_len,
              Tref(NEGATIVE_CUTOFF_VAL));

    if(is_softmax_applied)
        for(int j = 0; j < max_time_step * batch_size; j++)
            subvec_logsoftmax_cpu(&(probs[0]),
                                  &(workspace_cpu[problog_offset]),
                                  j * class_sz,
                                  j * class_sz,
                                  class_sz);
    else
        std::copy(probs.begin(), probs.end(), workspace_cpu.begin() + problog_offset);

    for(int j = 0; j < batch_size; j++)
    {
        int input_len     = workspace_cpu[j];
        int label_len     = workspace_cpu[batch_size + j];
        int label_offsets = workspace_cpu[2 * batch_size + j];
        int label_repeat  = workspace_cpu[3 * batch_size + j];
        auto lab_begin    = &(workspace_cpu[4 * batch_size]) + label_offsets;
        std::vector<int> indiv_lab(lab_begin, lab_begin + label_len);

        int alpha_offset_j    = alpha_offset + j * max_time_step * max_S_len;
        int lb_prime_offset_j = lb_prime_offset + j * max_S_len;
        ctc_alpha_cpu(probsDesc,
                      &(workspace_cpu[problog_offset]),
                      &(workspace_cpu[4 * batch_size + label_offsets]),
                      label_len,
                      input_len,
                      class_sz,
                      j,
                      label_repeat,
                      &(workspace_cpu[lb_prime_offset_j]),
                      &(workspace_cpu[alpha_offset_j]),
                      &(losses_cpu[j]),
                      blank_lb);

        ctc_gradient_cpu(probsDesc,
                         gradientsDesc,
                         &(workspace_cpu[lb_prime_offset_j]),
                         label_len,
                         input_len,
                         class_sz,
                         j,
                         label_repeat,
                         &(workspace_cpu[alpha_offset_j]),
                         &(workspace_cpu[problog_offset]),
                         &(gradients_cpu[0]),
                         blank_lb,
                         is_softmax_applied);
    }
}

template <typename Tgpu, typename Tref = Tgpu>
void VerifyCTCLoss(std::vector<int>& probsDesc,
                   std::vector<Tgpu>& probs,
                   const int* labels,
                   const int* labelLengths,
                   const int* inputLengths,
                   std::vector<Tref>& losses_cpu,
                   std::vector<int>& gradientsDesc,
                   std::vector<Tref>& gradients_cpu,
                   std::vector<Tref>& workspace_cpu,
                   const int blank_lb      = 0,
                   bool is_softmax_applied = true)
{
    if(probsDesc[0] != gradientsDesc[0] || probsDesc[1] != gradientsDesc[1] ||
       probsDesc[2] != gradientsDesc[2])
    {
        std::cout << "probs tensor's dimension does not gradients tensor's dimension" << std::endl;
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
            std::cout << "Wrong input time step at batch :" << i << std::endl;
            return;
        }
        max_label_len = std::max(max_label_len, labelLengths[i]);
        total_label_len += labelLengths[i];
        labels_offset[i] = i == 0 ? 0 : (labels_offset[i - 1] + labelLengths[i - 1]);

        for(int j = 0; j < labelLengths[i]; j++)
        {
            if(labels[labels_offset[i] + j] >= class_sz)
            {
                std::cout << "Wrong label id at batch :" << i << std::endl;
                return;
            }
            if(j > 0)
                if(labels[labels_offset[i] + j] == labels[labels_offset[i] + j - 1])
                    repeat[i]++;
        }

        if(labelLengths[i] + repeat[i] > inputLengths[i])
        {
            std::cout << "Error: label length exceeds input time step at batch :" << i << std::endl;
            return;
        }
    }

    if(probs.size() != (max_time_step * batch_size * class_sz))
    {
        std::cout << "Wrong probability tensor size" << std::endl;
        return;
    }
    if(probs.size() != gradients_cpu.size())
    {
        std::cout << "Wrong gradient tensor size" << std::endl;
        return;
    }

    // input length
    std::copy(inputLengths, inputLengths + batch_size, workspace_cpu.begin());

    // label length
    std::copy(labelLengths, labelLengths + batch_size, workspace_cpu.begin() + batch_size);

    // label offset
    std::copy(labels_offset.begin(),
              labels_offset.begin() + batch_size,
              workspace_cpu.begin() + 2 * batch_size);

    // label repeat
    std::copy(repeat.begin(), repeat.begin() + batch_size, workspace_cpu.begin() + 3 * batch_size);

    // labels
    std::copy(labels, labels + total_label_len, workspace_cpu.begin() + 4 * batch_size);

    launchCTCLoss(class_sz,
                  batch_size,
                  max_time_step,
                  max_label_len,
                  total_label_len,
                  probsDesc,
                  gradientsDesc,
                  probs,
                  losses_cpu,
                  gradients_cpu,
                  workspace_cpu,
                  blank_lb,
                  is_softmax_applied);
}

template <typename T>
void GetCTCLossWorkspaceSizeCPU(std::vector<int> probsDesc,
                                std::vector<int> gradientsDesc,
                                const int* labels,
                                const int* labelLengths,
                                const int* inputLengths,
                                size_t* workSpaceSizeCPU)
{
    if(probsDesc[0] != gradientsDesc[0] || probsDesc[1] != gradientsDesc[1] ||
       probsDesc[2] != gradientsDesc[2])
    {
        *workSpaceSizeCPU = 0;
        std::cout << "Label batch size does not match input batch size" << std::endl;
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
            std::cout << "Wrong input time step" << std::endl;
            return;
        }
        max_label_len = std::max(max_label_len, labelLengths[i]);
        total_label_len += labelLengths[i];
        labels_offset[i] = i == 0 ? 0 : (labels_offset[i - 1] + labelLengths[i - 1]);

        for(int j = 0; j < labelLengths[i]; j++)
        {
            if(labels[labels_offset[i] + j] >= class_sz)
            {
                std::cout << "Wrong label id at batc" << std::endl;
                return;
            }
            if(j > 0)
                if(labels[labels_offset[i] + j] == labels[labels_offset[i] + j - 1])
                    repeat[i]++;
        }

        if(labelLengths[i] + repeat[i] > inputLengths[i])
        {
            std::cout << "Error: label length exceeds input time step" << std::endl;
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

template <class T>
struct verify_ctcloss
{
    tensor<T> probs;
    std::vector<int> labels;
    std::vector<int> labelLengths;
    std::vector<int> inputLengths;
    tensor<T> losses;
    tensor<T> grads;

    miopen::CTCLossDescriptor ctcLossDesc;

    verify_ctcloss(const miopen::CTCLossDescriptor pCLD,
                   const tensor<T>& pPB,
                   const std::vector<int>& pLB,
                   const std::vector<int>& pLL,
                   const std::vector<int>& pIL,
                   const tensor<T>& pLS,
                   const tensor<T>& pGD)
    {
        ctcLossDesc  = pCLD;
        probs        = pPB;
        labels       = pLB;
        labelLengths = pLL;
        inputLengths = pIL;
        losses       = pLS;
        grads        = pGD;
    }

    std::tuple<tensor<T>, tensor<T>> cpu() const
    {
        int pdim0, pdim1, pdim2, pstr0, pstr1, pstr2;
        int gdim0, gdim1, gdim2, gstr0, gstr1, gstr2;
        std::tie(pstr0, pstr1, pstr2) = miopen::tien<3>(probs.desc.GetStrides());
        std::tie(pdim0, pdim1, pdim2) = miopen::tien<3>(probs.desc.GetLengths());
        std::tie(gstr0, gstr1, gstr2) = miopen::tien<3>(grads.desc.GetStrides());
        std::tie(gdim0, gdim1, gdim2) = miopen::tien<3>(grads.desc.GetLengths());

        std::vector<int> probsDim = {pdim0, pdim1, pdim2, pstr0, pstr1, pstr2};
        std::vector<int> gradsDim = {gdim0, gdim1, gdim2, gstr0, gstr1, gstr2};

        size_t workSpaceSizeCPU = 0;
        GetCTCLossWorkspaceSizeCPU<T>(probsDim,
                                      gradsDim,
                                      labels.data(),
                                      labelLengths.data(),
                                      inputLengths.data(),
                                      &workSpaceSizeCPU);

        size_t workSpaceDimCPU = workSpaceSizeCPU / sizeof(float);
        auto workSpace         = tensor<float>{workSpaceDimCPU};

        auto probs_cpu  = probs;
        auto losses_cpu = tensor<float>{losses.data.size()};
        auto grads_cpu  = tensor<float>{grads.data.size()};

        VerifyCTCLoss<T, float>(probsDim,
                                probs_cpu.data,
                                labels.data(),
                                labelLengths.data(),
                                inputLengths.data(),
                                losses_cpu.data,
                                gradsDim,
                                grads_cpu.data,
                                workSpace.data,
                                ctcLossDesc.blank_label_id,
                                ctcLossDesc.apply_softmax_layer);

        auto losses_T = tensor<T>{losses.data.size()};
        auto grads_T  = tensor<T>{grads.data.size()};

        for(size_t i         = 0; i < losses.data.size(); i++)
            losses_T.data[i] = T(losses_cpu.data[i]);

        for(size_t i        = 0; i < grads.data.size(); i++)
            grads_T.data[i] = T(grads_cpu.data[i]);

        auto retSet = std::make_tuple(losses_T, grads_T);
        return retSet;
    }

    std::tuple<tensor<T>, tensor<T>> gpu() const
    {
        auto&& handle = get_handle();

        size_t workSpaceSize = ctcLossDesc.GetCTCLossWorkspaceSize(handle,
                                                                   probs.desc,
                                                                   grads.desc,
                                                                   labels.data(),
                                                                   labelLengths.data(),
                                                                   inputLengths.data(),
                                                                   miopenCTCLossAlgo_t(0));

        auto workSpace     = tensor<T>{workSpaceSize / sizeof(T)};
        auto workSpace_dev = handle.Write(workSpace.data);

        auto losses_gpu = losses;
        auto grads_gpu  = grads;

        auto probs_dev  = handle.Write(probs.data);
        auto grads_dev  = handle.Write(grads_gpu.data);
        auto losses_dev = handle.Write(losses_gpu.data);

        ctcLossDesc.CTCLoss(handle,
                            probs.desc,
                            probs_dev.get(),
                            labels.data(),
                            labelLengths.data(),
                            inputLengths.data(),
                            losses_dev.get(),
                            grads.desc,
                            grads_dev.get(),
                            miopenCTCLossAlgo_t(0),
                            workSpace_dev.get(),
                            workSpaceSize);

        losses_gpu.data = handle.Read<T>(losses_dev, losses_gpu.data.size());
        grads_gpu.data  = handle.Read<T>(grads_dev, grads_gpu.data.size());

        auto retSet = std::make_tuple(losses_gpu, grads_gpu);
        return retSet;
    }

    void fail(int badtensor) const
    {
        std::cout << "CTC Loss: " << std::endl;
        std::cout << "Max Timestep, Batch Size, Number of Class: " << probs.desc.ToString()
                  << std::endl;

        switch(badtensor)
        {
        case(0): std::cout << "Losses" << std::endl; break;
        case(1): std::cout << "Gradients" << std::endl; break;
        default: break;
        }
    }
};

template <class T>
struct ctc_driver : test_driver
{
    int inputLen{};
    int labelLen{};
    int numClass{};
    int batchSize{};
    bool is_softmax_applied{};
    int blank_id{};

    miopen::CTCLossDescriptor ctcLossDesc;
    tensor<T> probs;
    tensor<T> grads;
    tensor<T> losses;

    ctc_driver()
    {
        add(batchSize, "batch-size", generate_data({1, 16, 32, 64, 128}));
        add(inputLen, "input-len", generate_data({100}));
        add(labelLen, "label-len", generate_data({40}));
        add(numClass, "num-class", generate_data({28, 5000}));
        add(is_softmax_applied, "apply-softmax-layer", generate_data({true, false}));
        add(blank_id, "blank-label-id", generate_data({0, 1000}));
    }

    void run()
    {
        if(type != miopenFloat)
            return;

        /// \todo Resolve the issue and remove workaround.
        /// The matching test cases fail on Jenkins from time to time.
        if(numClass == 5000 && is_softmax_applied)
            return;

        ctcLossDesc.dataType            = miopenFloat;
        ctcLossDesc.apply_softmax_layer = is_softmax_applied;
        ctcLossDesc.blank_label_id      = blank_id;

        std::vector<int> inputLengths(batchSize, inputLen);
        std::vector<int> labelLengths(batchSize, labelLen);

        for(int i           = 0; i < batchSize; i++)
            inputLengths[i] = GET_RAND() % inputLen + 1;

        for(int i           = 0; i < batchSize; i++)
            labelLengths[i] = GET_RAND() % labelLen + 1;

        for(int i = 0; i < batchSize; i++)
            if(inputLengths[i] < labelLengths[i] * 2 + 1)
                inputLengths[i] = labelLengths[i] * 2 + 1;

        int batch_sz      = batchSize;
        int class_sz      = numClass + 1;
        int max_time_step = *std::max_element(inputLengths.begin(), inputLengths.end());

        std::vector<int> probsDims  = {max_time_step, batch_sz, class_sz};
        std::vector<int> lossesDims = {batch_sz};

        unsigned long max_value = 17;

        probs = tensor<T>{probsDims}.generate(tensor_elem_gen_integer{max_value});
        for(int j = 0; j < batch_sz * max_time_step; j++)
        {
            T sum = T(0);
            for(int i = 0; i < class_sz; i++)
                sum += probs.data[j * class_sz + i];

            for(int i = 0; i < class_sz; i++)
                probs.data[j * class_sz + i] /= sum;
        }

        grads = tensor<T>{probsDims};
        std::fill(grads.begin(), grads.end(), T(0));

        losses = tensor<T>{lossesDims};
        std::fill(losses.begin(), losses.end(), T(0));

        size_t labels_sz = std::accumulate(labelLengths.begin(), labelLengths.end(), 0);

        auto labels  = std::vector<int>(labels_sz);
        int blank_lb = ctcLossDesc.blank_label_id;
        for(size_t i = 0; i < labels_sz; i++)
        {
            labels[i] = static_cast<int>(GET_RAND() % numClass + 1);
            if(blank_lb > numClass)
                labels[i] = labels[i] == numClass ? numClass - 1 : labels[i];
            else if(blank_lb < 0)
                labels[i] = labels[i] == 0 ? 1 : labels[i];
            else if(labels[i] == blank_lb)
                labels[i] = blank_lb - 1 >= 0 ? (blank_lb - 1) : blank_lb + 1;
        }

        verify(verify_ctcloss<T>{
            ctcLossDesc, probs, labels, labelLengths, inputLengths, losses, grads});
    }
};

int main(int argc, const char* argv[]) { test_drive<ctc_driver>(argc, argv); }
