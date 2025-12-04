// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#endif
#include "float_types.h"

constexpr static FLOAT negative_cutoff_val = -1e20;

inline __device__ FLOAT LogAddExp(const FLOAT* x, const FLOAT* y)
{
    FLOAT a = max(*x, *y);
    FLOAT b = min(*x, *y);
    FLOAT c = b - a;

    return c <= negative_cutoff_val ? max(a, negative_cutoff_val)
                                    // We don't need the extra precision of log1pf() and it adds
                                    // performance overhead.
                                    // cppcheck-suppress unpreciseMathCall
                                    : max(a + logf(expf(c) + 1), negative_cutoff_val);
}

template <class T>
inline __device__ void NonAtomicLogAddExp(const unsigned int& local_id,
                                          const unsigned int& label_length,
                                          const int* label_prime,
                                          const unsigned int& reverse_input_idx,
                                          const unsigned int& batch_id,
                                          const int& input_idx,
                                          T* beta_buff0,
                                          T* beta_buff1,
                                          const unsigned int& label_prime_len,
                                          T* alpha_log,
                                          T* gradients)
{
    __syncthreads();

    // TODO: https://github.com/ROCm/rocm-libraries/issues/2866
    // Consider parallelizing loop over multple threads rather than only using a couple of threads
    // in the group. There may also be a race condition where two threads (lid 0 and 1) may write to
    // the same gradients[gidx] location when label_cur values collide.
    if(local_id == 0 || local_id == 1)
    {
        for(unsigned int label_idx = 0; label_idx < label_length; label_idx++)
        {
            unsigned int label_offset = 2 * label_idx + local_id;
            unsigned const label_cur  = (local_id == 0) ? BLANK_LB : *(label_prime + label_offset);
            size_t gidx = reverse_input_idx * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + label_cur;
            T beta_temp =
                (input_idx % 2) == 0 ? *(beta_buff0 + label_offset) : *(beta_buff1 + label_offset);
            size_t bidx_ts = reverse_input_idx * label_prime_len + label_offset;

            beta_temp += *(alpha_log + bidx_ts);
            T grad_temp = gradients[gidx];

            gradients[gidx] = LogAddExp(&grad_temp, &beta_temp);
        }
    }
    if(local_id == 0)
    {
        unsigned int label_offset = 2 * label_length;
        size_t gidx = reverse_input_idx * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + BLANK_LB;
        T beta_temp =
            (input_idx % 2) == 0 ? *(beta_buff0 + label_offset) : *(beta_buff1 + label_offset);
        size_t bidx_ts = reverse_input_idx * label_prime_len + label_offset;

        beta_temp += *(alpha_log + bidx_ts);
        T grad_temp = gradients[gidx];

        gradients[gidx] = LogAddExp(&grad_temp, &beta_temp);
    }
}

template <class T>
inline __device__ void AtomicLogAddExp(const unsigned int& reverse_input_idx,
                                       const unsigned int& batch_id,
                                       const unsigned int& lb_cur,
                                       const unsigned int& label_prime_len,
                                       const unsigned int& reverse_label_idx,
                                       const T* alpha_log,
                                       T* gradients,
                                       T& beta_temp)
{
    static_assert(false && "Method only implemented for FP32");
}

template <>
inline __device__ void AtomicLogAddExp(const unsigned int& reverse_input_idx,
                                       const unsigned int& batch_id,
                                       const unsigned int& lb_cur,
                                       const unsigned int& label_prime_len,
                                       const unsigned int& reverse_label_idx,
                                       const float* alpha_log,
                                       float* gradients,
                                       float& beta_temp)
{
    size_t gidx    = reverse_input_idx * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + lb_cur;
    size_t bidx_ts = reverse_input_idx * label_prime_len + reverse_label_idx;
    beta_temp += *(alpha_log + bidx_ts);

    float* addr = gradients + gidx;
    float prev_val, cur_val = *addr;
    do
    {
        prev_val = cur_val;

        float a       = max(prev_val, beta_temp);
        float b       = min(prev_val, beta_temp);
        float c       = b - a;
        float new_val = c <= negative_cutoff_val ? max(a, negative_cutoff_val)
                                                 // We don't need the extra precision of log1pf()
                                                 // and it adds performance overhead.
                                                 // cppcheck-suppress unpreciseMathCall
                                                 : max(a + logf(expf(c) + 1), negative_cutoff_val);

        cur_val = atomicCAS(addr, prev_val, new_val);
    } while(cur_val != prev_val);
}

inline __device__ void CTCAlpha(const FLOAT* probs_logits,
                                const int* label_prime,
                                const unsigned int label_length,
                                const unsigned int input_length,
                                const unsigned int batch_id,
                                const unsigned int label_repeat,
                                FLOAT* alpha,
                                FLOAT* loss)
{
    const unsigned int local_id         = threadIdx.x;
    const unsigned int label_idx_offset = ((label_length + label_repeat) < input_length) ? 0 : 1;
    for(unsigned int i = label_idx_offset + local_id; i <= 1; i += WORK_PER_GRP)
    {
        unsigned int lb_cur = (i % 2 == 0) ? BLANK_LB : *(label_prime + i);
        unsigned int pidx   = batch_id * PROBS_STRIDE1 + lb_cur;
        *(alpha + i)        = *(probs_logits + pidx);
    }
    __syncthreads();

    const unsigned int label_prime_len = 2 * label_length + 1;
    for(unsigned int input_idx = 1; input_idx < input_length; input_idx++)
    {
        for(unsigned int label_idx = local_id; label_idx <= label_prime_len - 1;
            label_idx += WORK_PER_GRP)
        {
            unsigned int lb_cur = (label_idx % 2 == 0) ? BLANK_LB : *(label_prime + label_idx);
            unsigned int lb_pre = (label_idx % 2 == 0) ? BLANK_LB : *(label_prime + label_idx - 2);
            size_t pidx         = input_idx * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + lb_cur;
            size_t aidx_ts      = input_idx * label_prime_len + label_idx;
            size_t aidx_t1s     = (input_idx - 1) * label_prime_len + label_idx;

            FLOAT alpha_t1s1 = *(alpha + aidx_t1s - 1);
            FLOAT alpha_t1s  = *(alpha + aidx_t1s);

            FLOAT alpha_ts = (label_idx == 0) ? alpha_t1s : LogAddExp(&alpha_t1s, &alpha_t1s1);
            if(label_idx >= 2)
            {
                if(lb_cur != BLANK_LB && lb_cur != lb_pre)
                {
                    FLOAT alpha_t1s2 = *(alpha + aidx_t1s - 2);
                    alpha_ts         = LogAddExp(&alpha_ts, &alpha_t1s2);
                }
            }

            alpha_ts += *(probs_logits + pidx);
            *(alpha + aidx_ts) = max(alpha_ts, negative_cutoff_val);
        }
        __syncthreads();
    }

    if(local_id == 0)
    {
        unsigned int alpha_size = input_length * label_prime_len;
        FLOAT alp0              = *(alpha + alpha_size - 1);
        FLOAT alp1              = *(alpha + alpha_size - 2);
        *loss                   = -LogAddExp(&alp0, &alp1);
    }
}

inline __device__ void CTCGradient(const FLOAT* probs_logits,
                                   const int* label_prime,
                                   const unsigned int label_length,
                                   const unsigned int input_length,
                                   const unsigned int batch_id,
                                   const unsigned int label_repeat,
                                   FLOAT* alpha_log,
                                   FLOAT* beta_buff0,
                                   FLOAT* beta_buff1,
                                   const FLOAT* loss,
                                   FLOAT* gradients)
{
    const unsigned int local_id = threadIdx.x;
    for(unsigned int input_idx = 0; input_idx < input_length; input_idx++)
    {
        for(unsigned int i = local_id; i < CLASS_SZ; i += WORK_PER_GRP)
        {
            *(gradients + input_idx * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i) =
                negative_cutoff_val;
        }
    }
    __syncthreads();

    const unsigned int label_prime_len  = 2 * label_length + 1;
    const unsigned int label_idx_offset = ((label_length + label_repeat) < input_length) ? 0 : 1;
    for(unsigned int label_idx = label_idx_offset + local_id; label_idx <= 1;
        label_idx += WORK_PER_GRP)
    {
        unsigned int reverse_label_idx = label_prime_len - 1 - label_idx;
        unsigned int lb_cur =
            (reverse_label_idx % 2 == 0) ? BLANK_LB : *(label_prime + reverse_label_idx);
        unsigned int pidx = (input_length - 1) * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + lb_cur;
        unsigned int gidx = (input_length - 1) * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + lb_cur;
        unsigned int bidx_ts = (input_length - 1) * label_prime_len + reverse_label_idx;

        FLOAT probs_logits_pidx           = *(probs_logits + pidx);
        *(beta_buff0 + reverse_label_idx) = probs_logits_pidx;

        FLOAT alpha_temp = *(alpha_log + bidx_ts);
        alpha_temp += probs_logits_pidx;
        FLOAT grad_temp = negative_cutoff_val;

        gradients[gidx] = LogAddExp(&grad_temp, &alpha_temp);
    }
    __syncthreads();

    const FLOAT prob_lx_log = -*(loss);
    for(unsigned int i = local_id; i < CLASS_SZ; i += WORK_PER_GRP)
    {
        unsigned int pidx       = (input_length - 1) * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + i;
        unsigned int gidx       = (input_length - 1) * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i;
        FLOAT probs_logits_pidx = *(probs_logits + pidx);
        FLOAT grad_temp         = gradients[gidx];
        if constexpr(SOFTMAX_APPLIED == 0)
        {
            grad_temp -= probs_logits_pidx * 2;
        }
        else
        {
            grad_temp -= probs_logits_pidx;
        }
        grad_temp -= prob_lx_log;
        grad_temp = grad_temp <= negative_cutoff_val ? 0 : expf(grad_temp);

        if constexpr(SOFTMAX_APPLIED == 1)
        {
            *(gradients + gidx) = expf(probs_logits_pidx) - grad_temp;
        }
        else
        {
            *(gradients + gidx) = -grad_temp;
        }
    }
    __syncthreads();

    for(unsigned int input_idx = 1; input_idx < input_length; input_idx++)
    {
        unsigned int reverse_input_idx = input_length - 1 - input_idx;
        for(int label_idx = local_id; label_idx <= label_prime_len - 1; label_idx += WORK_PER_GRP)
        {
            int reverse_label_idx = label_prime_len - 1 - label_idx;
            int lb_cur =
                (reverse_label_idx % 2 == 0) ? BLANK_LB : *(label_prime + reverse_label_idx);
            int lb_pre =
                (reverse_label_idx % 2 == 0) ? BLANK_LB : *(label_prime + reverse_label_idx + 2);

            size_t pidx = reverse_input_idx * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + lb_cur;

            FLOAT beta_temp = (input_idx % 2 == 0) ? *(beta_buff1 + reverse_label_idx)
                                                   : *(beta_buff0 + reverse_label_idx);

            if(reverse_label_idx <= label_prime_len - 2)
            {
                FLOAT beta_temp1 = (input_idx % 2 == 0) ? *(beta_buff1 + reverse_label_idx + 1)
                                                        : *(beta_buff0 + reverse_label_idx + 1);
                beta_temp        = LogAddExp(&beta_temp, &beta_temp1);
            }
            if(reverse_label_idx <= (label_prime_len - 3))
            {
                if(lb_cur != BLANK_LB && lb_cur != lb_pre)
                {
                    FLOAT beta_temp2 = (input_idx % 2) == 0 ? *(beta_buff1 + reverse_label_idx + 2)
                                                            : *(beta_buff0 + reverse_label_idx + 2);
                    beta_temp        = LogAddExp(&beta_temp, &beta_temp2);
                }
            }

            beta_temp += *(probs_logits + pidx);
            beta_temp = max(beta_temp, negative_cutoff_val);
            if(input_idx % 2 == 0)
            {
                *(beta_buff0 + reverse_label_idx) = beta_temp;
            }
            else
            {
                *(beta_buff1 + reverse_label_idx) = beta_temp;
            }

            if constexpr(OPT_ATOMIC_LOGADDEXP == 1)
            {
                AtomicLogAddExp(reverse_input_idx,
                                batch_id,
                                lb_cur,
                                label_prime_len,
                                reverse_label_idx,
                                alpha_log,
                                gradients,
                                beta_temp);
            }
        }

        if constexpr(OPT_ATOMIC_LOGADDEXP == 0)
        {
            NonAtomicLogAddExp(local_id,
                               label_length,
                               label_prime,
                               reverse_input_idx,
                               batch_id,
                               input_idx,
                               beta_buff0,
                               beta_buff1,
                               label_prime_len,
                               alpha_log,
                               gradients);
        }

        __syncthreads();

        for(int i = local_id; i < CLASS_SZ; i += WORK_PER_GRP)
        {
            size_t pidx = reverse_input_idx * PROBS_STRIDE0 + batch_id * PROBS_STRIDE1 + i;
            size_t gidx = reverse_input_idx * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i;

            FLOAT probs_logits_pidx = *(probs_logits + pidx);

            FLOAT grad_temp = gradients[gidx];

            if constexpr(SOFTMAX_APPLIED == 0)
            {
                grad_temp -= probs_logits_pidx * 2;
            }
            else
            {
                grad_temp -= probs_logits_pidx;
            }

            grad_temp -= prob_lx_log;
            grad_temp = grad_temp <= negative_cutoff_val ? 0 : expf(grad_temp);

            if constexpr(SOFTMAX_APPLIED == 1)
            {
                *(gradients + gidx) = expf(probs_logits_pidx) - grad_temp;
            }
            else
            {
                *(gradients + gidx) = -grad_temp;
            }
        }
        __syncthreads();
    }
}

template <bool OptLclMemBeta, bool OptLclMemLb>
__forceinline__ __device__ void CTCLossImpl(const unsigned int local_id,
                                            const unsigned int group_id,
                                            const FLOAT* probs_logits,
                                            FLOAT* workspace,
                                            int* dim_data,
                                            FLOAT* losses,
                                            FLOAT* gradients,
                                            FLOAT* beta0  = nullptr,
                                            FLOAT* beta1  = nullptr,
                                            int* lb_prime = nullptr)
{
    for(unsigned int bid = group_id; bid < BATCH_SZ; bid += GRP_NUM)
    {
        unsigned int input_len     = *(dim_data + bid);
        unsigned int label_len     = *(dim_data + BATCH_SZ + bid);
        unsigned int label_offsets = *(dim_data + 2 * BATCH_SZ + bid);
        unsigned int label_repeat  = *(dim_data + 3 * BATCH_SZ + bid);

        if constexpr(OptLclMemLb)
        {
            for(unsigned int i = local_id; i < label_len; i += WORK_PER_GRP)
            {
                lb_prime[2 * i + 1] = dim_data[4 * BATCH_SZ + label_offsets + i];
            }
        }
        else
        {
            for(unsigned int i = local_id; i < label_len; i += WORK_PER_GRP)
            {
                dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN + 2 * i + 1] =
                    dim_data[4 * BATCH_SZ + label_offsets + i];
            }
        }

        for(unsigned int i = local_id; i < MAX_TSTEP * MAX_S_LEN; i += WORK_PER_GRP)
        {
            *(workspace + ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN + i) = negative_cutoff_val;
        }

        if constexpr(!OptLclMemBeta)
        {
            for(unsigned int i = local_id; i < 2 * MAX_S_LEN; i += WORK_PER_GRP)
            {
                *(workspace + BETA_OFFSET + bid * 2 * MAX_S_LEN + i) = negative_cutoff_val;
            }
        }

        __syncthreads();

        const int* label_prime;
        if constexpr(OptLclMemLb)
        {
            label_prime = lb_prime;
        }
        else
        {
            label_prime = &dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN];
        }
        CTCAlpha(probs_logits,
                 label_prime,
                 label_len,
                 input_len,
                 bid,
                 label_repeat,
                 &workspace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                 &losses[bid]);

        if constexpr(OptLclMemBeta)
        {
            for(unsigned int i = local_id; i < MAX_S_LEN; i += WORK_PER_GRP)
            {
                *(beta0 + i) = negative_cutoff_val;
                *(beta1 + i) = negative_cutoff_val;
            }
        }

        __syncthreads();

        FLOAT *beta_buf0, *beta_buf1;
        if constexpr(OptLclMemBeta)
        {
            beta_buf0 = beta0;
            beta_buf1 = beta1;
        }
        else
        {
            beta_buf0 = &workspace[BETA_OFFSET + bid * 2 * MAX_S_LEN];
            beta_buf1 = &workspace[BETA_OFFSET + (bid * 2 + 1) * MAX_S_LEN];
        }
        CTCGradient(probs_logits,
                    label_prime,
                    label_len,
                    input_len,
                    bid,
                    label_repeat,
                    &workspace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
                    beta_buf0,
                    beta_buf1,
                    &losses[bid],
                    gradients);
    }
}

template <bool OptLclMemBeta, bool OptLclMemLb>
__forceinline__ __device__ void CTCLoss(const unsigned int local_id,
                                        const unsigned int group_id,
                                        const FLOAT* probs_logits,
                                        FLOAT* workspace,
                                        int* dim_data,
                                        FLOAT* losses,
                                        FLOAT* gradients);

template <>
__forceinline__ __device__ void CTCLoss<false, false>(const unsigned int local_id,
                                                      const unsigned int group_id,
                                                      const FLOAT* probs_logits,
                                                      FLOAT* workspace,
                                                      int* dim_data,
                                                      FLOAT* losses,
                                                      FLOAT* gradients)
{
    CTCLossImpl<false, false>(
        local_id, group_id, probs_logits, workspace, dim_data, losses, gradients);
}

template <>
__forceinline__ __device__ void CTCLoss<true, false>(const unsigned int local_id,
                                                     const unsigned int group_id,
                                                     const FLOAT* probs_logits,
                                                     FLOAT* workspace,
                                                     int* dim_data,
                                                     FLOAT* losses,
                                                     FLOAT* gradients)
{
    __shared__ FLOAT beta0[MAX_S_LEN];
    __shared__ FLOAT beta1[MAX_S_LEN];
    CTCLossImpl<true, false>(
        local_id, group_id, probs_logits, workspace, dim_data, losses, gradients, beta0, beta1);
}

template <>
__forceinline__ __device__ void CTCLoss<false, true>(const unsigned int local_id,
                                                     const unsigned int group_id,
                                                     const FLOAT* probs_logits,
                                                     FLOAT* workspace,
                                                     int* dim_data,
                                                     FLOAT* losses,
                                                     FLOAT* gradients)
{
    __shared__ int lb_prime[MAX_S_LEN];
    CTCLossImpl<false, true>(local_id,
                             group_id,
                             probs_logits,
                             workspace,
                             dim_data,
                             losses,
                             gradients,
                             nullptr,
                             nullptr,
                             lb_prime);
}

template <>
__forceinline__ __device__ void CTCLoss<true, true>(const unsigned int local_id,
                                                    const unsigned int group_id,
                                                    const FLOAT* probs_logits,
                                                    FLOAT* workspace,
                                                    int* dim_data,
                                                    FLOAT* losses,
                                                    FLOAT* gradients)
{
    __shared__ FLOAT beta0[MAX_S_LEN];
    __shared__ FLOAT beta1[MAX_S_LEN];
    __shared__ int lb_prime[MAX_S_LEN];
    CTCLossImpl<true, true>(local_id,
                            group_id,
                            probs_logits,
                            workspace,
                            dim_data,
                            losses,
                            gradients,
                            beta0,
                            beta1,
                            lb_prime);
}

extern "C" __global__ void CTCLossGPU([[maybe_unused]] const FLOAT* probs,
                                      FLOAT* workspace,
                                      int* dim_data,
                                      FLOAT* losses,
                                      FLOAT* gradients)
{
    const FLOAT* probs_logits;
    if constexpr(SOFTMAX_APPLIED == 1)
    {
        probs_logits = &workspace[PROBLOG_OFFSET];
    }
    else
    {
        probs_logits = probs;
    }

    const unsigned int local_id = threadIdx.x;
    const unsigned int group_id = blockIdx.x;
    CTCLoss<OPT_LCL_MEM_BETA, OPT_LCL_MEM_LB>(
        local_id, group_id, probs_logits, workspace, dim_data, losses, gradients);
}
