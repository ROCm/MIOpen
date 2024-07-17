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

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

#define MIOPEN_USE_AMDGCN 0
#if defined(__AMDGCN__) && !(MIO_BN_GFX103X || MIO_BN_GFX110X || MIO_BN_GFX120X)
#undef MIOPEN_USE_AMDGCN
#define MIOPEN_USE_AMDGCN 1
#endif

#include "batchnorm_functions.h"
#include "reduction_functions.h"

#ifndef MIO_LAYOUT_NHWC
#define MIO_LAYOUT_NHWC 0
#endif

#if(MIO_LAYOUT_NHWC != 0) && (MIO_LAYOUT_NHWC != 1)
#error "MIO_LAYOUT_NHWC must be 0 or 1"
#endif

#if(MIO_BN_VARIANT == 0)

#define MIO_BN_SEGTMP_1 (MIO_BN_GRP0 / MIO_BN_HW)
#define MIO_BN_SEGTMP_2 ((MIO_BN_SEGTMP_1 == 0) ? 1 : MIO_BN_SEGTMP_1)
#define MIO_BN_SEGTMP (MIO_BN_HW * MIO_BN_SEGTMP_2)
#define MIO_BN_SEGMENT ((MIO_BN_SEGTMP > MIO_BN_NHW) ? (MIO_BN_NHW) : (MIO_BN_SEGTMP))
#define MIO_BN_NLOOP ((MIO_BN_NHW + MIO_BN_SEGMENT - 1) / MIO_BN_SEGMENT)
#define MIO_BN_SEGIHW (MIO_BN_SEGMENT / MIO_BN_HW)
#define MIO_BN_NLOOPM (MIO_BN_NLOOP - 1)
#define MIO_BN_SNHW (MIO_BN_NLOOPM * MIO_BN_SEGIHW)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatial(const __global _FLOAT* __restrict in,
                               __global _FLOAT* __restrict out,
                               __constant _FLOAT_PREC* __restrict scale,
                               __constant _FLOAT_PREC* __restrict bias,
                               _FLOAT_PREC INHW,
#if(MIO_RUNNING_RESULT == 1)
                               double expAvgFactor,
                               __global _FLOAT_PREC* __restrict resultRunningMean,
                               __global _FLOAT_PREC* __restrict resultRunningVariance,
#endif
                               double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                               ,
                               __global _FLOAT_PREC* __restrict resultSaveMean,
                               __global _FLOAT_PREC* __restrict resultSaveInvVariance
#endif
)
{

    // SPATIAL
    _FLOAT_ACCUM mean        = (_FLOAT_ACCUM)0.;
    _FLOAT_ACCUM variance    = (_FLOAT_ACCUM)0.;
    _FLOAT_ACCUM invVariance = (_FLOAT_ACCUM)0.;
    _FLOAT_ACCUM pvscale     = (_FLOAT_ACCUM)0.;
    _FLOAT_ACCUM pvbias      = (_FLOAT_ACCUM)0.;
    _FLOAT batchvalues[MIO_BN_NLOOP];
    _FLOAT_ACCUM temp;

    __local _FLOAT_PREC lcl_bias;
    __local _FLOAT_PREC lcl_scale;

    unsigned int index  = 0;
    unsigned int lid    = get_local_id(0);
    unsigned int grpid  = get_group_id(0);
    unsigned int chwid  = grpid * MIO_BN_HW + (lid % MIO_BN_HW);
    unsigned int lidihw = lid / MIO_BN_HW;
    unsigned int nid    = 0;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid < MIO_BN_SEGMENT)
    {
#if MIOPEN_USE_FP16 == 1
        __attribute__((opencl_unroll_hint(2)))
#endif
        for(unsigned int n = 0; n < MIO_BN_NLOOPM; ++n)
        {
            nid            = n * MIO_BN_SEGIHW + lidihw;
            index          = nid * MIO_BN_CHW + chwid;
            batchvalues[n] = (*(in + index));
            temp           = (_FLOAT_ACCUM)(*(in + index));
            mean += temp;
            variance = mad(temp, temp, variance);
        }
        nid                        = MIO_BN_SNHW + lidihw;
        index                      = nid * MIO_BN_CHW + chwid;
        batchvalues[MIO_BN_NLOOPM] = (index < MIO_BN_NCHW) ? (*(in + index)) : (_FLOAT)0.;
        temp                       = (_FLOAT_ACCUM)batchvalues[MIO_BN_NLOOPM];
        mean += temp;
        variance = mad(temp, temp, variance);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDS_SIZE];
    lds_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#else
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#endif

    variance = mad(-mean, mean, variance);
    if(variance < 0)
    {
        variance = 0;
    }
    invVariance = rsqrt(variance + (_FLOAT_ACCUM)epsilon);
    pvscale     = (_FLOAT_ACCUM)lcl_scale;
    pvbias      = (_FLOAT_ACCUM)lcl_bias;

    if(lid < MIO_BN_SEGMENT)
    {
        //==== CALC NORM =======================
        _FLOAT_ACCUM inhat = (_FLOAT_ACCUM)0.;

#if MIOPEN_USE_FP16 == 1
        __attribute__((opencl_unroll_hint(2)))
#endif
        for(unsigned int n = 0; n < MIO_BN_NLOOPM; n++)
        { // apply normalization
            inhat      = (batchvalues[n] - mean) * invVariance;
            nid        = n * MIO_BN_SEGIHW + lidihw;
            index      = nid * MIO_BN_CHW + chwid;
            out[index] = (_FLOAT)mad(pvscale, inhat, pvbias);
        } // end for

        // Tail of loop
        inhat = (batchvalues[MIO_BN_NLOOPM] - mean) * invVariance;
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        if(index < MIO_BN_NCHW)
            out[index] = (_FLOAT)mad(pvscale, inhat, pvbias);
    }

    if(lid == 0)
    {
#if(MIO_RUNNING_RESULT == 1)
        running_stash(
            resultRunningMean, resultRunningVariance, expAvgFactor, mean, variance, grpid);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        saved_stash(resultSaveMean, resultSaveInvVariance, mean, invVariance, grpid);
#endif
    }
} // end spatial norm

#elif(MIO_BN_VARIANT == 1)

//===========

#if MIO_LAYOUT_NHWC
#define MIO_MAX_READ 1
#define RD_BLK 1
#define GRPRD (MIO_BN_GRP0 * RD_BLK)
#else
#if(MIO_BN_HW >= 4096)
#define MIO_MAX_READ 3
#else
#define MIO_MAX_READ 2
#endif
#define RD_BLK 1
#define GRPRD (MIO_BN_GRP0 * RD_BLK * 4)
#endif

#define MIO_BN_REM4 (MIO_BN_NHW - ((MIO_BN_NHW / GRPRD) * GRPRD))
#define MIO_BN_LESS4 (MIO_BN_NHW - MIO_BN_REM4)
#define MIO_BN_CHUNK4 (MIO_MAX_READ * GRPRD)
#define MIO_BN_REMOUT4 (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_CHUNK4) * MIO_BN_CHUNK4))
#define MIO_BN_LESSOUT4 (MIO_BN_NHW - MIO_BN_REMOUT4)
#define MIO_BN_REM (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_GRP0) * MIO_BN_GRP0))
#define MIO_BN_LESS (MIO_BN_NHW - MIO_BN_REM)
#define MIO_BN_CHUNK (MIO_MAX_READ * MIO_BN_GRP0)
#define MIO_BN_REMOUT (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_CHUNK) * MIO_BN_CHUNK))
#define MIO_BN_LESSOUT (MIO_BN_NHW - MIO_BN_REMOUT)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatial(const __global _FLOAT* __restrict in,
                               __global _FLOAT* __restrict out,
                               __constant _FLOAT_PREC* __restrict scale,
                               __constant _FLOAT_PREC* __restrict bias,
                               _FLOAT_PREC INHW,
#if(MIO_RUNNING_RESULT == 1)
                               double expAvgFactor,
                               __global _FLOAT_PREC* __restrict resultRunningMean,
                               __global _FLOAT_PREC* __restrict resultRunningVariance,
#endif
                               double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                               ,
                               __global _FLOAT_PREC* __restrict resultSaveMean,
                               __global _FLOAT_PREC* __restrict resultSaveInvVariance
#endif
)
{

    // SPATIAL

    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
    _FLOAT_PREC variance    = (_FLOAT_PREC)0.;
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pvscale, pvbias;

    __local _FLOAT_PREC lcl_bias;
    __local _FLOAT_PREC lcl_scale;

    uint index = 0;
    uint lid   = get_local_id(0);
    uint grpid = get_group_id(0);
#if !MIO_LAYOUT_NHWC
    uint chwid = grpid * MIO_BN_HW;
#endif
    uint nidx  = 0;
    uint hwidx = 0;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if !MIO_LAYOUT_NHWC && MIO_BN_HW >= 4096
    _FLOAT4 read4;
    __attribute__((opencl_unroll_hint(2))) for(unsigned int k = lid << 2; k < MIO_BN_LESS4;
                                               k += GRPRD)
    {
        nidx  = k / MIO_BN_HW;
        hwidx = k - (nidx * MIO_BN_HW);
        index = nidx * MIO_BN_CHW + chwid + hwidx;
        read4 = *((const global _FLOAT4*)(in + index));
        mean += (_FLOAT_PREC)read4.x;
        mean += (_FLOAT_PREC)read4.y;
        mean += (_FLOAT_PREC)read4.z;
        mean += (_FLOAT_PREC)read4.w;
        variance = mad((_FLOAT_PREC)read4.x, (_FLOAT_PREC)read4.x, variance);
        variance = mad((_FLOAT_PREC)read4.y, (_FLOAT_PREC)read4.y, variance);
        variance = mad((_FLOAT_PREC)read4.z, (_FLOAT_PREC)read4.z, variance);
        variance = mad((_FLOAT_PREC)read4.w, (_FLOAT_PREC)read4.w, variance);
    }

#if(MIO_BN_REM4)
    unsigned int remkey = (lid << 2) + MIO_BN_LESS4;
    nidx                = remkey / MIO_BN_HW;
    hwidx               = remkey - (nidx * MIO_BN_HW);
    index               = nidx * MIO_BN_CHW + chwid + hwidx;
    if(index < (MIO_BN_NCHW - 3))
    {
        read4 = *((const global _FLOAT4*)(in + index));
        mean += (_FLOAT_PREC)read4.x;
        mean += (_FLOAT_PREC)read4.y;
        mean += (_FLOAT_PREC)read4.z;
        mean += (_FLOAT_PREC)read4.w;
        variance = mad((_FLOAT_PREC)read4.x, (_FLOAT_PREC)read4.x, variance);
        variance = mad((_FLOAT_PREC)read4.y, (_FLOAT_PREC)read4.y, variance);
        variance = mad((_FLOAT_PREC)read4.z, (_FLOAT_PREC)read4.z, variance);
        variance = mad((_FLOAT_PREC)read4.w, (_FLOAT_PREC)read4.w, variance);
    }

#endif

#else
    __attribute__((opencl_unroll_hint(4))) for(unsigned int k = lid; k < MIO_BN_LESS;
                                               k += MIO_BN_GRP0)
    {
        nidx            = k / MIO_BN_HW;
        hwidx           = k - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
        index           = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
#else
        index = nidx * MIO_BN_CHW + chwid + hwidx;
#endif
        _FLOAT_PREC xin = (_FLOAT_PREC)(*(in + index));
        mean += xin;
        variance = mad(xin, xin, variance);
    }
#if(MIO_BN_REM)
    if(lid < MIO_BN_REM)
    {
        unsigned int remkey = lid + MIO_BN_LESS;
        nidx                = remkey / MIO_BN_HW;
        hwidx               = remkey - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
        index               = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
#else
        index = nidx * MIO_BN_CHW + chwid + hwidx;
#endif
        _FLOAT_PREC xin = (index < MIO_BN_NCHW) ? (_FLOAT_PREC)(*(in + index)) : (_FLOAT_PREC)0.;
        mean += xin;
        variance = mad(xin, xin, variance);
    }
#endif
#endif
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDS_SIZE];
    lds_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#else
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#endif

    // REDUCTION COMPLETE ---------------------------
    variance = mad(-mean, mean, variance);
    if(variance < 0)
    {
        variance = 0;
    }
    invVariance = rsqrt(variance + epsilon);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

#if(MIO_LAYOUT_NHWC || MIO_BN_REM == 0)
    const unsigned int k_limit =
#if MIO_LAYOUT_NHWC
        MIO_BN_NHW;
#else
        MIO_BN_LESS;
#endif
    __attribute__((opencl_unroll_hint(2))) for(unsigned int k = lid; k < k_limit; k += MIO_BN_GRP0)
    {
        nidx  = k / MIO_BN_HW;
        hwidx = k - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
        index = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
#else
        index = nidx * MIO_BN_CHW + chwid + hwidx;
#endif
        out[index] =
            (_FLOAT)mad(pvscale, ((_FLOAT_PREC)(*(in + index)) - mean) * invVariance, pvbias);
    } // end for
#else
    _FLOAT_PREC xhat[MIO_MAX_READ];
    __attribute__((opencl_unroll_hint(2))) for(unsigned int k = (MIO_MAX_READ * lid);
                                               k < MIO_BN_LESSOUT;
                                               k += MIO_BN_CHUNK)
    {
        for(unsigned int j = 0; j < MIO_MAX_READ; j++)
        {
            unsigned int l = k + j;
            nidx           = l / MIO_BN_HW;
            hwidx          = l - (nidx * MIO_BN_HW);
            index          = nidx * MIO_BN_CHW + chwid + hwidx;
            xhat[j]        = ((_FLOAT_PREC)(*(in + index)) - mean) * invVariance;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for(unsigned int j = 0; j < MIO_MAX_READ; j++)
        {
            unsigned int l = k + j;
            nidx           = l / MIO_BN_HW;
            hwidx          = l - (nidx * MIO_BN_HW);
            index          = nidx * MIO_BN_CHW + chwid + hwidx;
            *(out + index) = (_FLOAT)mad(pvscale, xhat[j], pvbias);
        }
    } // end for

#if(MIO_BN_REMOUT)
    unsigned int remkeyout = (MIO_MAX_READ * lid) + MIO_BN_LESSOUT;
    for(unsigned int j = 0; j < MIO_MAX_READ; j++)
    {
        unsigned int l  = remkeyout + j;
        nidx            = l / MIO_BN_HW;
        hwidx           = l - (nidx * MIO_BN_HW);
        index           = nidx * MIO_BN_CHW + chwid + hwidx;
        _FLOAT_PREC xin = (index < MIO_BN_NCHW) ? (_FLOAT_PREC)(*(in + index)) : (_FLOAT_PREC)0.;
        xhat[j]         = (xin - mean) * invVariance;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    for(unsigned int j = 0; j < MIO_MAX_READ; j++)
    {
        unsigned int l = remkeyout + j;
        nidx           = l / MIO_BN_HW;
        hwidx          = l - (nidx * MIO_BN_HW);
        index          = nidx * MIO_BN_CHW + chwid + hwidx;
        if(index < MIO_BN_NCHW)
        {
            *(out + index) = (_FLOAT)mad(pvscale, xhat[j], pvbias);
        }
    }
#endif
#endif

    if(lid == 0)
    {
#if(MIO_RUNNING_RESULT == 1)
        running_stash(
            resultRunningMean, resultRunningVariance, expAvgFactor, mean, variance, grpid);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        saved_stash(resultSaveMean, resultSaveInvVariance, mean, invVariance, grpid);
#endif
    }
} // end spatial norm

#elif(MIO_BN_VARIANT == 2) // MULTI-KERNEL reduction for > 33M elements

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatialNorm(const __global _FLOAT* __restrict in,
                                   __global _FLOAT* __restrict out,
                                   const __global _FLOAT* __restrict scale,
                                   const __global _FLOAT* __restrict bias)
{

    // SPATIAL
    _FLOAT mean        = (_FLOAT)0.;
    _FLOAT invVariance = (_FLOAT)0.;
    _FLOAT inhat       = (_FLOAT)0.;
    _FLOAT pvt_scale   = (_FLOAT)0.;
    _FLOAT pvt_bias    = (_FLOAT)0.;
    __local _FLOAT lcl_mean, lcl_ivar, lcl_scale, lcl_bias;

    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx           = xgid * MIO_BN_HW;
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;

    // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
    if(get_local_id(1) == 0)
    {
        lcl_scale = *(scale + xgid);
        lcl_bias  = *(bias + xgid);
        lcl_mean  = *(out + meanstashindex); // load stashed mean
        lcl_ivar  = *(out + varstashindex);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
        mean        = lcl_mean;
        invVariance = lcl_ivar;
        pvt_scale   = lcl_scale;
        pvt_bias    = lcl_bias;
#if(MIO_BN_HW > MIO_BN_LOOP_UNROLL_MAXHW)
        for(unsigned int n = 0; n < MIO_BN_N; n++)
#else
        __attribute__((opencl_unroll_hint(2))) for(unsigned int n = 0; n < MIO_BN_N; n++)
#endif
        { // apply normalization
            index = n * MIO_BN_CHW + cidx + ygid;
            inhat = (*(in + index) - mean) * invVariance;
            // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
            out[index] = mad(pvt_scale, inhat, pvt_bias);
        } // end for(n)
    }     // end if(inImgIndex)
} // end spatial norm

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatialFinalMeanVariance(
    __global _FLOAT* __restrict meanvarbuff,
    _FLOAT INHW
#if(MIO_RUNNING_RESULT == 1)
    ,
    double expAvgFactor /* input momentum */
    ,
    __global _FLOAT* __restrict resultRunningMean, /*input and output*/
    __global _FLOAT* __restrict resultRunningVariance
#endif
    ,
    double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
    ,
    __global _FLOAT* __restrict resultSaveMean /*output only*/
    ,
    __global _FLOAT* __restrict resultSaveInvVariance
#endif
)
{
    _FLOAT variance             = (_FLOAT)0.;
    _FLOAT invVariance          = (_FLOAT)0.;
    _FLOAT mean                 = (_FLOAT)0.;
    unsigned int lid            = get_local_id(1);
    unsigned int ygrp_id        = get_group_id(1);
    unsigned int xgid           = get_global_id(0);
    unsigned int ygrp_sz        = get_local_size(1);
    unsigned int yngrps         = get_num_groups(1);
    unsigned int cidx           = xgid * MIO_BN_HW;
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;
    unsigned int commitID       = 0;

    for(int gn = 0; gn < yngrps; gn++)
    {
        unsigned int offset    = gn * ygrp_sz + lid;
        unsigned int meanindex = cidx + ygrp_sz * offset;
        unsigned int varindex  = cidx + ygrp_sz * offset + 2;
        if(offset < yngrps)
        { // modify to span larger number of groups
            mean += *(meanvarbuff + meanindex);
            variance += *(meanvarbuff + varindex); // load per group variance
        }
    }

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDS_SIZE];
    lds_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#else
    commitID = 64;
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#endif

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    variance = mad(-mean, mean, variance);
    if(variance < 0)
    {
        variance = 0;
    }
    invVariance = rsqrt(variance + epsilon);
    if(lid == commitID)
    {
        meanvarbuff[meanstashindex] = mean;        // stash mean
        meanvarbuff[varstashindex]  = invVariance; // stash mean
    }

    // Save mean and calculate and save running mean
    unsigned int ygid = get_global_id(1);
    if(ygid == commitID)
    {
#if(MIO_RUNNING_RESULT == 1)
        running_stash(resultRunningMean, resultRunningVariance, expAvgFactor, mean, variance, xgid);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        saved_stash(resultSaveMean, resultSaveInvVariance, mean, invVariance, xgid);
#endif
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatialMeanVariance(const __global _FLOAT* __restrict in,
                                           __global _FLOAT* __restrict mvbuff)
{

    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx      = xgid * MIO_BN_HW;
    unsigned int meanindex = cidx + ygrp_sz * ygrp_id;
    unsigned int varindex  = meanindex + 2;
    _FLOAT mean            = (_FLOAT)0.;
    _FLOAT variance        = (_FLOAT)0.;
    _FLOAT value           = (_FLOAT)0.;

    if(ygid < MIO_BN_HW)
    {
#if(MIO_BN_HW > MIO_BN_LOOP_UNROLL_MAXHW)
        for(unsigned int n = 0; n < MIO_BN_N; n++)
#else
        __attribute__((opencl_unroll_hint(2))) for(unsigned int n = 0; n < MIO_BN_N; n++)
#endif
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            value = *(in + index);
            mean += value;
            variance = mad(value, value, variance);
        }
    }

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDS_SIZE];
    lds_reduce2(&mean, &variance, (_FLOAT_ACCUM)1.0, lcl_data_x, lcl_data_y, ylid);
#else
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)1.0, lcl_data_x, lcl_data_y, ylid);
#endif

    if(ylid == 0)
    {
        mvbuff[meanindex] = mean;
        mvbuff[varindex]  = variance;
    }
} // end spatial mean kernel

#elif(MIO_BN_VARIANT == 3)

// This kernel implies the image is greater than a wavefront, but smaller than 257
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdTrainSpatial(const __global _FLOAT* __restrict in,
                               __global _FLOAT* __restrict out,
                               __constant _FLOAT_PREC* __restrict scale,
                               __constant _FLOAT_PREC* __restrict bias,
                               _FLOAT_PREC INHW,
#if(MIO_RUNNING_RESULT == 1)
                               double expAvgFactor,
                               __global _FLOAT_PREC* __restrict resultRunningMean,
                               __global _FLOAT_PREC* __restrict resultRunningVariance,
#endif
                               double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                               ,
                               __global _FLOAT_PREC* __restrict resultSaveMean,
                               __global _FLOAT_PREC* __restrict resultSaveInvVariance
#endif
)
{

    // SPATIAL
    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
    _FLOAT_PREC variance    = (_FLOAT_PREC)0.;
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC inhat       = (_FLOAT_PREC)0.;
    _FLOAT_PREC pvscale     = (_FLOAT_PREC)0.;
    _FLOAT_PREC pvbias      = (_FLOAT_PREC)0.;
    _FLOAT_PREC xin         = (_FLOAT_PREC)0.;

    local _FLOAT_PREC lcl_bias;
    local _FLOAT_PREC lcl_scale;

    unsigned int index = 0;
    unsigned int lid   = get_local_id(0);
    unsigned int grpid = get_group_id(0);
    unsigned int cidx  = grpid * MIO_BN_HW;

#if(MIO_BN_N < MIO_BN_MAXN)
    _FLOAT minibatch[MIO_BN_N];
#endif

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }

    if(lid < MIO_BN_HW)
    {
        __attribute__((opencl_unroll_hint(2))) for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + lid;
            xin   = (_FLOAT_PREC)(*(in + index));
            mean += xin;
            variance     = mad(xin, xin, variance);

#if(MIO_BN_N < MIO_BN_MAXN)
            minibatch[n] = (*(in + index));
#endif
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDS_SIZE];
    lds_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#else
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#endif

    variance = mad(-mean, mean, variance);
    if(variance < 0)
    {
        variance = 0;
    }
    invVariance = rsqrt(variance + (_FLOAT_PREC)epsilon);

    if(lid < MIO_BN_HW)
    {
        pvscale = lcl_scale;
        pvbias  = lcl_bias;

        __attribute__((opencl_unroll_hint(2))) for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index      = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            inhat      = (minibatch[n] - mean) * invVariance; // (in[index] - mean) * invVariance;
#else
            inhat = ((_FLOAT_PREC)(*(in + index)) - mean) * invVariance;
#endif
            out[index] = (_FLOAT)mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

    if(lid == 0)
    {
#if(MIO_RUNNING_RESULT == 1)
        running_stash(
            resultRunningMean, resultRunningVariance, expAvgFactor, mean, variance, grpid);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        saved_stash(resultSaveMean, resultSaveInvVariance, mean, invVariance, grpid);
#endif
    }
} // end spatial norm

#elif(MIO_BN_VARIANT == 4)

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#endif
// Batch size 1 and 2
/* __attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2)))  */
__kernel void MIOpenBatchNormFwdTrainSpatial(const __global _FLOAT* __restrict in,
                                             __global _FLOAT* __restrict out,
                                             __constant _FLOAT_PREC* __restrict scale,
                                             __constant _FLOAT_PREC* __restrict bias,
                                             _FLOAT_PREC INHW,
#if(MIO_RUNNING_RESULT == 1)
                                             double expAvgFactor,
                                             __global _FLOAT_PREC* __restrict resultRunningMean,
                                             __global _FLOAT_PREC* __restrict resultRunningVariance,
#endif
                                             double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                                             ,
                                             __global _FLOAT_PREC* __restrict resultSaveMean,
                                             __global _FLOAT_PREC* __restrict resultSaveInvVariance
#endif
                                             ,
                                             unsigned int imageDims,
                                             unsigned int batchStride)
{

    unsigned int grpid = get_group_id(0);
    unsigned int lid   = get_local_id(0);
    unsigned int lsz   = get_local_size(0);

    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
    _FLOAT_PREC variance    = (_FLOAT_PREC)0.;
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pvscale     = (_FLOAT_PREC)0.;
    _FLOAT_PREC pvbias      = (_FLOAT_PREC)0.;
    _FLOAT_PREC xin         = (_FLOAT_PREC)0.;

    unsigned int index0 = 0;
#if(MIO_BN_N == 2)
    unsigned int index1 = 0;
#endif

    local _FLOAT_PREC lcl_bias;
    local _FLOAT_PREC lcl_scale;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }

    unsigned int cidx = grpid * imageDims;
    for(int idx = lid; idx < imageDims; idx += lsz)
    {
        index0 = cidx + idx;
        xin    = (_FLOAT_PREC)(*(in + index0));
        mean += xin;
        variance = mad(xin, xin, variance);
#if(MIO_BN_N == 2)
        index0 += batchStride;
        xin = (_FLOAT_PREC)(*(in + index0));
        mean += xin;
        variance = mad(xin, xin, variance);
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDS_SIZE];
    lds_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#else
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#endif

    variance    = mad(-mean, mean, variance);
    variance    = variance > 0. ? variance : 0.;
    invVariance = rsqrt(variance + (_FLOAT_PREC)epsilon);
    pvscale     = lcl_scale;
    pvbias      = lcl_bias;

    for(int idx = lid; idx < imageDims; idx += lsz)
    {
        index0             = cidx + idx;
#if(MIO_BN_N == 2)
        index1             = batchStride + index0;
#endif
        _FLOAT_PREC inhat0 = ((_FLOAT_PREC)(*(in + index0)) - mean) * invVariance;
#if(MIO_BN_N == 2)
        _FLOAT_PREC inhat1 = ((_FLOAT_PREC)(*(in + index1)) - mean) * invVariance;
#endif
        out[index0]        = (_FLOAT)mad(pvscale, inhat0, pvbias);
#if(MIO_BN_N == 2)
        out[index1]        = (_FLOAT)mad(pvscale, inhat1, pvbias);
#endif
    }

    if(lid == 0)
    {
#if(MIO_RUNNING_RESULT == 1)
        running_stash_dyn(
            resultRunningMean, resultRunningVariance, expAvgFactor, mean, variance, grpid, INHW);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        saved_stash(resultSaveMean, resultSaveInvVariance, mean, invVariance, grpid);
#endif
    }
} // end spatial norm

#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif

#endif

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
