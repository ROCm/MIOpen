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

#ifndef USE_HIP_BACKEND
#define USE_HIP_BACKEND 0
#endif

#ifndef USE_OCL_BACKEND
#define USE_OCL_BACKEND 0
#endif

#if USE_OCL_BACKEND == 1
#if defined(cl_khr_int64_base_atomics) && defined(cl_khr_int64_extended_atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#define ATOM64
#endif

#if __OPENCL_VERSION__ > CL_VERSION_1_0
#define ATOM32
#elif defined(cl_khr_global_int32_base_atomics) && defined(cl_khr_global_int32_extended_atomics)
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL_EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL_EXTENSION cl_khr_local_int32_extended_atomics : enable
#define ATOM32
#else
#error "Required integer atomics not supported by this OpenCL implemenation."
#endif
#endif

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#ifndef NEGATIVE_CUTOFF_VAL
#define NEGATIVE_CUTOFF_VAL (_FLOAT)(-1e20)
#endif

#ifndef SOFTMAX_LEN
#define SOFTMAX_LEN 1
#endif

#ifndef SOFTMAX_APPLIED
#define SOFTMAX_APPLIED 1
#endif

#ifndef PROBS_STRIDE0
#define PROBS_STRIDE0 (BATCH_SZ * CLASS_SZ)
#endif
#ifndef PROBS_STRIDE1
#define PROBS_STRIDE1 CLASS_SZ
#endif

#if SOFTMAX_APPLIED == 1
#define USE_PROBS_STRIDE0 (BATCH_SZ * CLASS_SZ)
#define USE_PROBS_STRIDE1 CLASS_SZ
#else
#define USE_PROBS_STRIDE0 PROBS_STRIDE0
#define USE_PROBS_STRIDE1 PROBS_STRIDE1
#endif

#ifndef GRADS_STRIDE0
#define GRADS_STRIDE0 (BATCH_SZ * CLASS_SZ)
#endif
#ifndef GRADS_STRIDE1
#define GRADS_STRIDE1 CLASS_SZ
#endif

#ifndef BLANK_LB_ID
#define BLANK_LB 0
#elif BLANK_LB_ID < 0
#define BLANK_LB 0
#elif BLANK_LB_ID >= CLASS_SZ
#define BLANK_LB (CLASS_SZ - 1)
#else
#define BLANK_LB BLANK_LB_ID
#endif

#ifdef OPT_LCL_MEM_GRAD
#define ADDRSPACE_GRAD __local
#else
#define ADDRSPACE_GRAD __global
#endif

#ifdef OPT_LCL_MEM_LB
#define ADDRSPACE_LB __local
#else
#define ADDRSPACE_LB __global
#endif

#ifdef OPT_LCL_MEM_BETA
#define ADDRSPACE_BETA __local
#else
#define ADDRSPACE_BETA __global
#endif

static inline _FLOAT LogAddExp(const _FLOAT* x, const _FLOAT* y)
{
    _FLOAT a = max(*x, *y);
    _FLOAT b = min(*x, *y);
    _FLOAT c = b - a;

    return c <= NEGATIVE_CUTOFF_VAL ? max(a, NEGATIVE_CUTOFF_VAL)
                                    : max(a + log(exp(b - a) + 1), NEGATIVE_CUTOFF_VAL);
}

inline void AtomicLogAddExp(volatile ADDRSPACE_GRAD float* addr, const float operand)
{
    union
    {
        unsigned int uval;
        float fval;
    } newVal, curVal, prevVal, a, b, c;
    curVal.fval = *((ADDRSPACE_GRAD float*)addr);

    do
    {
        prevVal.fval = curVal.fval;

        a.fval      = max(prevVal.fval, operand);
        b.fval      = min(prevVal.fval, operand);
        c.fval      = b.fval - a.fval;
        newVal.fval = c.fval <= NEGATIVE_CUTOFF_VAL
                          ? max(a.fval, NEGATIVE_CUTOFF_VAL)
                          : max(a.fval + log(exp(b.fval - a.fval) + 1), NEGATIVE_CUTOFF_VAL);

        curVal.uval =
#ifdef ATOM64
            atom_cmpxchg(
#else
            atomic_cmpxchg(
#endif
                (volatile ADDRSPACE_GRAD unsigned int*)addr, prevVal.uval, newVal.uval);

    } while(curVal.uval != prevVal.uval);
}

static inline void CTCAlpha(const global _FLOAT* probs_logits,
                            const ADDRSPACE_LB int* label_prime,
                            const unsigned int label_length,
                            const unsigned int input_length,
                            const unsigned int batch_id,
                            const unsigned int label_repeat,
                            global _FLOAT* alpha,
                            global _FLOAT* loss)
{
    uint label_prime_len = 2 * label_length + 1;

    uint lid = get_local_id(0);

    uint aidx0 = label_length + label_repeat < input_length ? 0 : 1;
    uint aidx1 = 1;
    for(uint i = aidx0 + lid; i <= aidx1; i += WORK_PER_GRP)
    {
        uint lb_cur = i % 2 == 0 ? BLANK_LB : *((const ADDRSPACE_LB int*)(label_prime + i));
        uint pidx   = batch_id * USE_PROBS_STRIDE1 + lb_cur;
        *((global _FLOAT*)(alpha + i)) = *((const global _FLOAT*)(probs_logits + pidx));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint j = 1; j < input_length; j++)
    {

        for(uint i = lid; i <= label_prime_len - 1; i += WORK_PER_GRP)
        {
            uint lb_cur = i % 2 == 0 ? BLANK_LB : *((const ADDRSPACE_LB int*)(label_prime + i));
            uint lb_pre = i % 2 == 0 ? BLANK_LB : *((const ADDRSPACE_LB int*)(label_prime + i - 2));
            size_t pidx = j * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + lb_cur;
            size_t aidx_ts  = j * label_prime_len + i;
            size_t aidx_t1s = (j - 1) * label_prime_len + i;

            _FLOAT alpha_t1s1 = *((global _FLOAT*)(alpha + aidx_t1s - 1));
            _FLOAT alpha_t1s  = *((global _FLOAT*)(alpha + aidx_t1s));

            _FLOAT alpha_ts = i == 0 ? alpha_t1s : LogAddExp(&alpha_t1s, &alpha_t1s1);
            if(i >= 2)
                if(lb_cur != BLANK_LB && lb_cur != lb_pre)
                {
                    _FLOAT alpha_t1s2 = *((global _FLOAT*)(alpha + aidx_t1s - 2));
                    alpha_ts          = LogAddExp(&alpha_ts, &alpha_t1s2);
                }

            alpha_ts += *((global _FLOAT*)(probs_logits + pidx));
            *((global _FLOAT*)(alpha + aidx_ts)) = max(alpha_ts, NEGATIVE_CUTOFF_VAL);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0)
    {
        uint alpha_size         = input_length * label_prime_len;
        _FLOAT alp0             = *((global _FLOAT*)(alpha + alpha_size - 1));
        _FLOAT alp1             = *((global _FLOAT*)(alpha + alpha_size - 2));
        *((global _FLOAT*)loss) = -LogAddExp(&alp0, &alp1);
    }
}

static inline void CTCGradient(const global _FLOAT* probs_logits,
                               const ADDRSPACE_LB int* label_prime,
                               const unsigned int label_length,
                               const unsigned int input_length,
                               const unsigned int batch_id,
                               const unsigned int label_repeat,
                               global _FLOAT* alpha_log,
                               ADDRSPACE_BETA _FLOAT* beta_buff0,
                               ADDRSPACE_BETA _FLOAT* beta_buff1,
                               const global _FLOAT* loss,
                               global _FLOAT* gradients
#ifdef OPT_LCL_MEM_GRAD
                               ,
                               local _FLOAT* gradtmp
#endif
)
{
    uint label_prime_len = 2 * label_length + 1;

    _FLOAT prob_lx_log = -*((const global _FLOAT*)loss);

    uint lid = get_local_id(0);

    uint aidx0 = 1;
    uint aidx1 = label_length + label_repeat < input_length ? 0 : 1;

    for(uint j = 0; j < input_length; j++)
        for(uint i = lid; i < CLASS_SZ; i += WORK_PER_GRP)
        {
            *((global _FLOAT*)(gradients + j * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i)) =
                NEGATIVE_CUTOFF_VAL;
        }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint k = aidx1 + lid; k <= aidx0; k += WORK_PER_GRP)
    {
        uint k1     = label_prime_len - 1 - k;
        uint lb_cur = k1 % 2 == 0 ? BLANK_LB : *((const ADDRSPACE_LB int*)(label_prime + k1));
        uint pidx = (input_length - 1) * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + lb_cur;
#ifndef OPT_LCL_MEM_GRAD
        uint gidx = (input_length - 1) * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + lb_cur;
#endif
        uint bidx_ts = (input_length - 1) * label_prime_len + k1;

        _FLOAT probs_logits_pidx = *((const global _FLOAT*)(probs_logits + pidx));
        *((ADDRSPACE_BETA _FLOAT*)(beta_buff0 + k1)) = probs_logits_pidx;

        _FLOAT alpha_temp = *((global _FLOAT*)(alpha_log + bidx_ts));
        alpha_temp += probs_logits_pidx;
        _FLOAT grad_temp = NEGATIVE_CUTOFF_VAL;

#ifdef OPT_LCL_MEM_GRAD
        gradtmp[lb_cur]
#else
        gradients[gidx]
#endif
            = LogAddExp(&grad_temp, &alpha_temp);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = lid; i < CLASS_SZ; i += WORK_PER_GRP)
    {
        uint pidx = (input_length - 1) * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + i;
        uint gidx = (input_length - 1) * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i;
        _FLOAT probs_logits_pidx = *((const global _FLOAT*)(probs_logits + pidx));
        _FLOAT grad_temp =
#ifdef OPT_LCL_MEM_GRAD
            gradtmp[i]
#else
            gradients[gidx]
#endif
            ;
        grad_temp -= probs_logits_pidx
#if SOFTMAX_APPLIED == 0
                     * 2
#endif
            ;
        grad_temp -= prob_lx_log;
        grad_temp = grad_temp <= NEGATIVE_CUTOFF_VAL ? 0 : exp(grad_temp);

        *((global _FLOAT*)(gradients + gidx)) =
#if SOFTMAX_APPLIED == 1
            exp(probs_logits_pidx)
#endif
            - grad_temp;

#ifdef OPT_LCL_MEM_GRAD
        gradtmp[i] = NEGATIVE_CUTOFF_VAL;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint j = 1; j < input_length; j++)
    {
        int j1 = input_length - 1 - j;

        for(int k = lid; k <= label_prime_len - 1; k += WORK_PER_GRP)
        {
            int k1     = label_prime_len - 1 - k;
            int lb_cur = k1 % 2 == 0 ? BLANK_LB : *((const ADDRSPACE_LB int*)(label_prime + k1));
            int lb_pre =
                k1 % 2 == 0 ? BLANK_LB : *((const ADDRSPACE_LB int*)(label_prime + k1 + 2));

            size_t pidx = j1 * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + lb_cur;

            _FLOAT beta_temp = j % 2 == 0 ? *((ADDRSPACE_BETA _FLOAT*)(beta_buff1 + k1))
                                          : *((ADDRSPACE_BETA _FLOAT*)(beta_buff0 + k1));

            if(k1 <= label_prime_len - 2)
            {
                _FLOAT beta_temp1 = j % 2 == 0 ? *((ADDRSPACE_BETA _FLOAT*)(beta_buff1 + k1 + 1))
                                               : *((ADDRSPACE_BETA _FLOAT*)(beta_buff0 + k1 + 1));
                beta_temp = LogAddExp(&beta_temp, &beta_temp1);
            }
            if(k1 <= label_prime_len - 3)
                if(lb_cur != BLANK_LB && lb_cur != lb_pre)
                {
                    _FLOAT beta_temp2 = j % 2 == 0
                                            ? *((ADDRSPACE_BETA _FLOAT*)(beta_buff1 + k1 + 2))
                                            : *((ADDRSPACE_BETA _FLOAT*)(beta_buff0 + k1 + 2));
                    beta_temp = LogAddExp(&beta_temp, &beta_temp2);
                }

            beta_temp += *((const global _FLOAT*)(probs_logits + pidx));
            beta_temp = max(beta_temp, NEGATIVE_CUTOFF_VAL);
            if(j % 2 == 0)
                *((ADDRSPACE_BETA _FLOAT*)(beta_buff0 + k1)) = beta_temp;
            else
                *((ADDRSPACE_BETA _FLOAT*)(beta_buff1 + k1)) = beta_temp;

#ifdef OPT_ATOMIC_LOGADDEXP
#ifndef OPT_LCL_MEM_GRAD
            size_t gidx = j1 * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + lb_cur;
#endif
            size_t bidx_ts = j1 * label_prime_len + k1;
            beta_temp += *((global _FLOAT*)(alpha_log + bidx_ts));

            AtomicLogAddExp((
#ifdef OPT_LCL_MEM_GRAD
                                local _FLOAT*)(gradtmp + lb_cur
#else
                                global _FLOAT*)(gradients + gidx
#endif
                                               ),
                            beta_temp);
#else
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(lid == 0 || lid == 1)
            for(int k = 0; k < label_length; k++)
            {
                int klid   = 2 * k + lid;
                int lb_cur = lid == 0 ? BLANK_LB : *((const ADDRSPACE_LB int*)(label_prime + klid));
#ifndef OPT_LCL_MEM_GRAD
                size_t gidx      = j1 * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + lb_cur;
#endif
                _FLOAT beta_temp = j % 2 == 0 ? *((ADDRSPACE_BETA _FLOAT*)(beta_buff0 + klid))
                                              : *((ADDRSPACE_BETA _FLOAT*)(beta_buff1 + klid));
                size_t bidx_ts = j1 * label_prime_len + klid;

                beta_temp += *((global _FLOAT*)(alpha_log + bidx_ts));
                _FLOAT grad_temp =
#ifdef OPT_LCL_MEM_GRAD
                    gradtmp[lb_cur]
#else
                    gradients[gidx]
#endif
                    ;

#ifdef OPT_LCL_MEM_GRAD
                gradtmp[lb_cur]
#else
                gradients[gidx]
#endif
                    = LogAddExp(&grad_temp, &beta_temp);
            }

        if(lid == 0)
        {
            int k            = 2 * label_length;

#ifndef OPT_LCL_MEM_GRAD
            size_t gidx      = j1 * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + BLANK_LB;
#endif
            _FLOAT beta_temp = j % 2 == 0 ? *((ADDRSPACE_BETA _FLOAT*)(beta_buff0 + k))
                                          : *((ADDRSPACE_BETA _FLOAT*)(beta_buff1 + k));
            size_t bidx_ts = j1 * label_prime_len + k;

            beta_temp += *((global _FLOAT*)(alpha_log + bidx_ts));
            _FLOAT grad_temp =
#ifdef OPT_LCL_MEM_GRAD
                gradtmp[BLANK_LB]
#else
                gradients[gidx]
#endif
                ;

#ifdef OPT_LCL_MEM_GRAD
            gradtmp[BLANK_LB]
#else
            gradients[gidx]
#endif
                = LogAddExp(&grad_temp, &beta_temp);
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int i = lid; i < CLASS_SZ; i += WORK_PER_GRP)
        {
            size_t pidx = j1 * USE_PROBS_STRIDE0 + batch_id * USE_PROBS_STRIDE1 + i;
            size_t gidx = j1 * GRADS_STRIDE0 + batch_id * GRADS_STRIDE1 + i;

            _FLOAT probs_logits_pidx = *((const global _FLOAT*)(probs_logits + pidx));

            _FLOAT grad_temp =
#ifdef OPT_LCL_MEM_GRAD
                gradtmp[i]
#else
                gradients[gidx]
#endif
                ;

            grad_temp -= probs_logits_pidx
#if SOFTMAX_APPLIED == 0
                         * 2
#endif
                ;
            grad_temp -= prob_lx_log;
            grad_temp = grad_temp <= NEGATIVE_CUTOFF_VAL ? 0 : exp(grad_temp);

            *((global _FLOAT*)(gradients + gidx)) =
#if SOFTMAX_APPLIED == 1
                exp(probs_logits_pidx)
#endif
                - grad_temp;

#ifdef OPT_LCL_MEM_GRAD
            *((local _FLOAT*)(gradtmp + i)) = NEGATIVE_CUTOFF_VAL;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

kernel void CTCLossGPU(const global _FLOAT* probs,
                       global _FLOAT* workSpace,
                       global int* dim_data,
                       global _FLOAT* losses,
                       global _FLOAT* gradients)
{

    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    uint grp_id = gid / WORK_PER_GRP;

#ifdef OPT_LCL_MEM_BETA
    local _FLOAT beta0[MAX_S_LEN];
    local _FLOAT beta1[MAX_S_LEN];
#endif

#ifdef OPT_LCL_MEM_GRAD
    local _FLOAT gradtmp[CLASS_SZ];
#endif

#ifdef OPT_LCL_MEM_LB
    local int lb_prime[MAX_S_LEN];
#endif

    for(uint bid = grp_id; bid < BATCH_SZ; bid += GRP_NUM)
    {
        uint input_len     = *((global int*)(dim_data + bid));
        uint label_len     = *((global int*)(dim_data + BATCH_SZ + bid));
        uint label_offsets = *((global int*)(dim_data + 2 * BATCH_SZ + bid));
        uint label_repeat  = *((global int*)(dim_data + 3 * BATCH_SZ + bid));

        for(uint i = lid; i < label_len; i += WORK_PER_GRP)
#ifdef OPT_LCL_MEM_LB
            lb_prime[
#else
            dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN +
#endif
                2 * i + 1] = dim_data[4 * BATCH_SZ + label_offsets + i];

        for(uint i = lid; i < MAX_TSTEP * MAX_S_LEN; i += WORK_PER_GRP)
            *((global _FLOAT*)(workSpace + ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN + i)) =
                NEGATIVE_CUTOFF_VAL;

#ifndef OPT_LCL_MEM_BETA
        for(uint i = lid; i < 2 * MAX_S_LEN; i += WORK_PER_GRP)
            *((global _FLOAT*)(workSpace + BETA_OFFSET + bid * 2 * MAX_S_LEN + i)) =
                NEGATIVE_CUTOFF_VAL;
#endif

        barrier(CLK_LOCAL_MEM_FENCE);

        CTCAlpha(
#if SOFTMAX_APPLIED == 1
            &workSpace[PROBLOG_OFFSET]
#else
            probs
#endif
            ,
#ifdef OPT_LCL_MEM_LB
            &lb_prime[0]
#else
            &dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN]
#endif
            ,
            label_len,
            input_len,
            bid,
            label_repeat,
            &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
            &losses[bid]);

#ifdef OPT_LCL_MEM_GRAD
        for(uint i = lid; i < CLASS_SZ; i += WORK_PER_GRP)
            *((local _FLOAT*)(gradtmp + i)) = NEGATIVE_CUTOFF_VAL;
#endif

#ifdef OPT_LCL_MEM_BETA
        for(uint i = lid; i < MAX_S_LEN; i += WORK_PER_GRP)
        {
            *((local _FLOAT*)(beta0 + i)) = NEGATIVE_CUTOFF_VAL;
            *((local _FLOAT*)(beta1 + i)) = NEGATIVE_CUTOFF_VAL;
        }
#endif

        barrier(CLK_LOCAL_MEM_FENCE);

        CTCGradient(
#if SOFTMAX_APPLIED == 1
            &workSpace[PROBLOG_OFFSET]
#else
            probs
#endif
            ,
#ifdef OPT_LCL_MEM_LB
            &lb_prime[0]
#else
            &dim_data[LB_PRIME_OFFSET + bid * MAX_S_LEN]
#endif
            ,
            label_len,
            input_len,
            bid,
            label_repeat,
            &workSpace[ALPHA_OFFSET + bid * MAX_TSTEP * MAX_S_LEN],
#ifdef OPT_LCL_MEM_BETA
            &beta0[0],
            &beta1[0]
#else
            &workSpace[BETA_OFFSET + bid * 2 * MAX_S_LEN],
            &workSpace[BETA_OFFSET + (bid * 2 + 1) * MAX_S_LEN]
#endif
            ,
            &losses[bid],
            gradients
#ifdef OPT_LCL_MEM_GRAD
            ,
            &gradtmp[0]
#endif
        );
    }

    (void)probs;
}
