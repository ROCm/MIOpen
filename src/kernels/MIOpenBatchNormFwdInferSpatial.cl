/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 1
#endif

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

#ifndef MIO_BN_GRP0
#define MIO_BN_GRP0 1
#endif

#ifndef MIO_BN_GRP1
#define MIO_BN_GRP1 1
#endif

#ifndef MIO_BN_GRP2
#define MIO_BN_GRP2 1
#endif

#ifndef MIO_BN_NGRPS
#define MIO_BN_NGRPS 1
#endif

#ifndef MIO_BN_VARIANT
#define MIO_BN_VARIANT 0
#endif

#define UNUSED __attribute__((__unused__))

#ifdef __AMDGCN__
#undef __AMDGCN__
#endif

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

static inline void ReduceKernel(__local _FLOAT* lcl_mem,
                                unsigned int sum_stride,
                                unsigned int unit_id,
                                unsigned int unit_len)
{
    _FLOAT sum              = 0;
    unsigned int lcl_offset = unit_id * unit_len;

#pragma unroll
    for(unsigned int i = 0; i < unit_len; i += sum_stride)
    {
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}

#if(MIO_BN_VARIANT == 0)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdInferSpatialEst(const __global _FLOAT* __restrict in, /* x input */
                            __global _FLOAT* __restrict out,      /* y output */
                            const __global _FLOAT* __restrict estimatedMean,
                            const __global _FLOAT* __restrict estimatedVariance,
                            const __global _FLOAT* __restrict scale,
                            const __global _FLOAT* __restrict bias,
                            double epsilon)
{

    int xgid = get_global_id(0);
    int ygid = get_global_id(1);

    local _FLOAT lmean;
    local _FLOAT lvar;
    local _FLOAT lscale;
    local _FLOAT lbias;

    unsigned int cidx = xgid * MIO_BN_HW;
    unsigned int index;

    _FLOAT mean, variance, invVariance;
    _FLOAT inhat;
    _FLOAT pscale, pbias;

    if(get_local_id(1) == 0)
    {
        lmean  = estimatedMean[xgid];
        lvar   = estimatedVariance[xgid];
        lscale = scale[xgid]; // dims 1xCx1x1
        lbias  = bias[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mean        = lmean;
    variance    = lvar;
    pscale      = lscale;
    pbias       = lbias;
    invVariance = rsqrt(fabs(variance + epsilon));

    // move across the sections of the image mini_batch stack

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index      = n * MIO_BN_CHW + cidx + ygid;
            inhat      = (in[index] - mean) * invVariance;
            out[index] = mad(pscale, inhat, pbias); // y_i = gamma*x_hat + beta
        }                                           // end for(img_offset)
    }
} // end spatial norm

//=========================================================

//=== SPATIAL NO SAVED DATA ===============================

#elif(MIO_BN_VARIANT == 1)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdInferSpatialSingleNorm(const __global _FLOAT* __restrict in,
                                   __global _FLOAT* __restrict out,
                                   const __global _FLOAT* __restrict scale,
                                   const __global _FLOAT* __restrict bias,
                                   double epsilon,
                                   double INHW)
{

    // SPATIAL
    _FLOAT mean        = 0.;
    _FLOAT variance    = 0.;
    _FLOAT invVariance = 0.;
    _FLOAT inhat       = 0.;
    _FLOAT pvt_scale   = 0.;
    _FLOAT pvt_bias    = 0.;
    _FLOAT elemStd     = 0.;

    local _FLOAT lscale;
    local _FLOAT lbias;

    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid * MIO_BN_HW;

    if(ylid == 0)
    {
        lscale = scale[xgid];
        lbias  = bias[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            mean += in[index];
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_GRP1 > 64)
    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    _FLOAT tmp     = 0.;
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[ylid];
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    if(ylid == 63)
        lcl_data[0] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0];

#elif(MIO_BN_GRP1 > 16)
    _FLOAT tmp = 0.;
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    mean = as_float(__builtin_amdgcn_readlane(as_int(mean), 63));

#else
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
    mean = as_float(__builtin_amdgcn_readlane(as_int(mean), 0));
#endif
#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_GRP1 > 16)
    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0];
#else

    mean = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        mean += lcl_data[i];
    }

#endif

#endif

    // if(ygid==0) printf("premean: %f\n" ,mean);
    mean *= INHW;
    // if(ygid==0) printf("postmean: %f\n" ,mean);
    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index   = n * MIO_BN_CHW + cidx + ygid;
            elemStd = (in[index] - mean);
            variance += elemStd * elemStd;
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_GRP1 > 64)
    tmp            = 0.;
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[ylid];
    barrier(CLK_LOCAL_MEM_FENCE);
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    if(ylid == 63)
        lcl_data[0] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0];

#elif(MIO_BN_GRP1 > 16)
    tmp  = 0.;
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance = as_float(__builtin_amdgcn_readlane(as_int(variance), 63));

#else
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance = as_float(__builtin_amdgcn_readlane(as_int(variance), 0));
#endif
    variance *= INHW;

#else

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
#if(MIO_BN_GRP1 > 16)
    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0] * INHW;
#else

    variance = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        variance += lcl_data[i];
    }
    variance *= INHW;
#endif
#endif

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    // #4 apply the normalization
    // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
    if(ygid < MIO_BN_HW)
    {

        pvt_scale = lscale;
        pvt_bias  = lbias;

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index = n * MIO_BN_CHW + cidx + ygid;
            inhat = (in[index] - mean) * invVariance;
            // #5 Gamma and Beta adjust
            // y_i = gamma*x_hat + beta
            out[index] = mad(pvt_scale, inhat, pvt_bias);
        } // end for
    }     // end if

} // end spatial norm

#elif(MIO_BN_VARIANT == 2)

__kernel void BatchNormFwdInferSpatialNorm(const __global _FLOAT* __restrict in,
                                           __global _FLOAT* __restrict out,
                                           const __global _FLOAT* __restrict scale,
                                           const __global _FLOAT* __restrict bias)
{

    // SPATIAL
    _FLOAT mean   = 0.;
    _FLOAT invVar = 0.;
    _FLOAT inhat  = 0.;
    _FLOAT pscale = 0.;
    _FLOAT pbias  = 0.;
    __local _FLOAT lb, ls, lm, liv;
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int ylid    = get_local_id(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;

    // #4 apply the normalization
    // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
    if(ylid == 0)
    {
        unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
        unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;
        lm                          = out[meanstashindex]; // load stashed mean
        liv                         = out[varstashindex];
        ls                          = scale[xgid];
        lb                          = bias[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
        pscale = ls;
        pbias  = lb;
        mean   = lm;
        invVar = liv;
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index = n * MIO_BN_CHW + cidx + ygid;
            inhat = (in[index] - mean) * invVar;
            out[index] =
                mad(pscale, inhat, pbias); // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
        }                                  // end for(n)
    }                                      // end if(inImgIndex)
} // end spatial norm

__kernel void BatchNormFwdInferSpatialFinalVariance(__global _FLOAT* __restrict varbuff,
                                                    double epsilon)
{

    // SPATIAL
    _FLOAT variance    = 0.;
    _FLOAT invVariance = 0.;

    unsigned int ylid          = get_local_id(1);
    unsigned int ygrp_id       = get_group_id(1);
    unsigned int xgid          = get_global_id(0);
    unsigned int ygrp_sz       = get_local_size(1);
    unsigned int yngrps        = get_num_groups(1);
    unsigned int cidx          = xgid * MIO_BN_HW;
    unsigned int varstashindex = cidx + ygrp_sz * ygrp_id + 3;
    _FLOAT NHW                 = (_FLOAT)MIO_BN_NHW;

    for(int gn = 0; gn < yngrps; gn++)
    {
        unsigned int offset   = gn * ygrp_sz + ylid;
        unsigned int varindex = cidx + ygrp_sz * offset + 2;
        if(offset < yngrps)
        {                                  // modified to span larger number of groups
            variance += varbuff[varindex]; // load per group variance
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_NGRPS > 64)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    _FLOAT tmp     = 0.;
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    variance = lcl_data[ylid];
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance /= NHW;
    invVariance = rsqrt(variance + epsilon);
    if(ylid == 63)
    {
        varbuff[varstashindex] = invVariance; // stash mean
    }

#elif(MIO_BN_NGRPS > 16)

    _FLOAT tmp = 0.;
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance = as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    variance /= NHW;
    invVariance = rsqrt(variance + epsilon);
    if(ylid == 63)
    {
        varbuff[varstashindex] = invVariance; // stash mean
    }

#else
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x108, 15, 15, 0));
    variance /= NHW;
    invVariance = rsqrt(variance + epsilon);
    if(ylid == 0)
    {
        varbuff[varstashindex] = invVariance; // stash mean
    }

#endif

#else
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
#if(MIO_BN_NGRPS > 16)
    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    variance    = lcl_data[0] / NHW;

#else

    variance = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        variance += lcl_data[i];
    }
    variance /= NHW;
#endif
    invVariance = rsqrt(variance + epsilon);
    if(ylid == 0)
    {
        varbuff[varstashindex] = invVariance; // stash mean
    }
#endif
}

__kernel void BatchNormFwdInferSpatialVariance(const __global _FLOAT* __restrict in,
                                               __global _FLOAT* __restrict meanvarbuff)
{

    // SPATIAL
    _FLOAT mean     = 0.;
    _FLOAT elemStd  = 0.;
    _FLOAT variance = 0.;

    __local _FLOAT lm;

    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index, ncIdx;
    unsigned int cidx     = xgid * MIO_BN_HW;
    unsigned int varindex = cidx + ygrp_sz * ygrp_id + 2;

    if(ylid == 0)
    {
        unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
        lm                          = meanvarbuff[meanstashindex]; // load stashed mean
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
        mean = lm;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            ncIdx   = n * MIO_BN_CHW + cidx;
            index   = ncIdx + ygid;
            elemStd = (in[index] - mean);
            variance += elemStd * elemStd;
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_GRP1 > 64)

    _FLOAT tmp = 0.;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    variance = lcl_data[ylid];
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    if(ylid == 63)
    {
        meanvarbuff[varindex] = variance; // pre-stage for group reduction
    }

#elif(MIO_BN_GRP1 > 16)

    _FLOAT tmp = 0.;
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    if(ylid == 63)
    {
        meanvarbuff[varindex] = variance; // pre-stage for group reduction
    }
#else
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x108, 15, 15, 0));
    if(ylid == 0)
    {
        meanvarbuff[varindex] = variance; // pre-stage for group reduction
    }

#endif

#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Reduction over a work-grp: 256 -> 64 -> 16 -> 1
    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
    {
        meanvarbuff[varindex] = lcl_data[0];
    }
#endif

} // end spatial variance

__kernel void BatchNormFwdInferSpatialFinalMean(__global _FLOAT* __restrict meanvarbuff)
{

    _FLOAT mean = 0.;

    unsigned int ylid           = get_local_id(1);
    unsigned int ygrp_id        = get_group_id(1);
    unsigned int xgid           = get_global_id(0);
    unsigned int ygrp_sz        = get_local_size(1);
    unsigned int yngrps         = get_num_groups(1);
    unsigned int cidx           = xgid * MIO_BN_HW;
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    _FLOAT NHW                  = (_FLOAT)MIO_BN_NHW;

    for(int gn = 0; gn < yngrps; gn++)
    {
        unsigned int offset    = gn * ygrp_sz + ylid;
        unsigned int meanindex = cidx + ygrp_sz * offset;
        if(offset < yngrps)
        { // modify to span larger number of groups
            mean += meanvarbuff[meanindex];
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_NGRPS > 64)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    _FLOAT tmp = 0.;
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    mean = lcl_data[ylid];
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    if(ylid == 63)
    {
        meanvarbuff[meanstashindex] = mean / NHW; // stash mean
    }
#elif(MIO_BN_NGRPS > 16)
    _FLOAT tmp = 0.;
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    if(ylid == 63)
    {
        meanvarbuff[meanstashindex] = mean / NHW; // stash mean
    }
#else
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x108, 15, 15, 0));
    if(ylid == 0)
    {
        meanvarbuff[meanstashindex] = mean / NHW; // stash mean
    }
#endif

#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
#if(MIO_BN_NGRPS > 16)
    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0] / NHW;
#else

    mean = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        mean += lcl_data[i];
    }
    mean /= NHW;
#endif
    if(ylid == 0)
    {
        meanvarbuff[meanstashindex] = mean; // stash mean
    }

#endif
}

__kernel void BatchNormFwdInferSpatialMean(const __global _FLOAT* __restrict in,
                                           __global _FLOAT* __restrict meanbuff)
{

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    _FLOAT mean            = 0.;
    unsigned int ylid      = get_local_id(1);
    unsigned int ygrp_id   = get_group_id(1);
    unsigned int xgid      = get_global_id(0);
    unsigned int ygid      = get_global_id(1);
    unsigned int ygrp_sz   = get_local_size(1);
    unsigned int index     = 0;
    unsigned int cidx      = xgid * MIO_BN_HW;
    unsigned int meanindex = cidx + ygrp_sz * ygrp_id; // making assumption of n=0 here

    // move across the sections of the image mini_batch stack
    if(ygid < MIO_BN_HW)
    {

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            mean += in[index];
        }
    }

    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__

    _FLOAT tmp = 0.;
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    mean = lcl_data[ylid];
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;

    if(ylid == 63)
    {
        meanbuff[meanindex] = mean; // pre-stage for group reduction
    }
#else

#if(MIO_BN_GRP1 > 16)
    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0];
#else

    mean = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        mean += lcl_data[i];
    }
#endif
    if(ylid == 0)
    {
        meanbuff[meanindex] = mean; // pre-stage for group reduction
    }
#endif

} // end spatial mean kernel

//====================================================

#endif

#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
