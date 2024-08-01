/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#include "activation_functions.h"
#include "reduction_functions.h"

#if(MIO_BN_VARIANT == 0)

#define MIO_BN_SEGTMP (MIO_BN_HW * (MIO_BN_GRP0 / MIO_BN_HW))
#define MIO_BN_SEGMENT ((MIO_BN_SEGTMP > MIO_BN_NHW) ? (MIO_BN_NHW) : (MIO_BN_SEGTMP))
#define MIO_BN_NLOOP ((MIO_BN_NHW + MIO_BN_SEGMENT - 1) / MIO_BN_SEGMENT)
#define MIO_BN_SEGIHW (MIO_BN_SEGMENT / MIO_BN_HW)
#define MIO_BN_NLOOPM (MIO_BN_NLOOP - 1)
#define MIO_BN_SNHW (MIO_BN_NLOOPM * MIO_BN_SEGIHW)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormActivFwdTrainSpatial(float INHW,
                                    const _FLOAT alpha,
                                    const _FLOAT beta,
                                    const _FLOAT gamma,
                                    double epsilon,
#if(MIO_RUNNING_RESULT == 1)
                                    double expAvgFactor,
#endif
                                    const __global _FLOAT* __restrict in,
                                    __global _FLOAT* __restrict out,
                                    __constant _FLOAT_PREC* __restrict bias,
                                    __constant _FLOAT_PREC* __restrict scale

#if(MIO_RUNNING_RESULT == 1)
                                    ,
                                    __global _FLOAT_PREC* __restrict runningMean,
                                    __global _FLOAT_PREC* __restrict runningVariance
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
                                    ,
                                    __global _FLOAT_PREC* __restrict savedInvVariance,
                                    __global _FLOAT_PREC* __restrict savedMean
#endif

)
{

    // SPATIAL
    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
    _FLOAT_PREC variance    = (_FLOAT_PREC)0.;
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pvscale     = (_FLOAT_PREC)0.;
    _FLOAT_PREC pvbias      = (_FLOAT_PREC)0.;
    _FLOAT_PREC batchvalues[MIO_BN_NLOOP];

    __local _FLOAT_PREC lcl_bias;
    __local _FLOAT_PREC lcl_scale;

    unsigned int index  = 0;
    unsigned int lid    = get_local_id(0);
    unsigned int grpid  = get_group_id(0);
    unsigned int chwid  = grpid * MIO_BN_HW + (lid % MIO_BN_HW);
    unsigned int lidihw = lid / MIO_BN_HW;
    unsigned int nid    = 0;
    _FLOAT_PREC bn_out  = 0.;
    _FLOAT_PREC act_out = 0.;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid < MIO_BN_SEGMENT)
    {
#if MIOPEN_USE_FP16 == 0
        __attribute__((opencl_unroll_hint(2)))
#endif
        for(unsigned int n = 0; n < MIO_BN_NLOOPM; ++n)
        {
            nid            = n * MIO_BN_SEGIHW + lidihw;
            index          = nid * MIO_BN_CHW + chwid;
            batchvalues[n] = (_FLOAT_PREC)(*(in + index));
            mean += batchvalues[n];
            variance = mad(batchvalues[n], batchvalues[n], variance);
        }
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        batchvalues[MIO_BN_NLOOPM] =
            (index < MIO_BN_NCHW) ? (_FLOAT_PREC)(*(in + index)) : (_FLOAT_PREC)0.;
        mean += batchvalues[MIO_BN_NLOOPM];
        variance = mad(batchvalues[MIO_BN_NLOOPM], batchvalues[MIO_BN_NLOOPM], variance);
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

    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + (_FLOAT_PREC)epsilon);
    pvscale     = (_FLOAT_PREC)lcl_scale;
    pvbias      = (_FLOAT_PREC)lcl_bias;

    if(lid < MIO_BN_SEGMENT)
    {
        //==== CALC NORM =======================
        _FLOAT_PREC inhat = (_FLOAT_PREC)0.;

        for(unsigned int n = 0; n < MIO_BN_NLOOPM; n++)
        { // apply normalization
            inhat  = (batchvalues[n] - (_FLOAT_PREC)mean) * ((_FLOAT_PREC)invVariance);
            nid    = n * MIO_BN_SEGIHW + lidihw;
            index  = nid * MIO_BN_CHW + chwid;
            bn_out = mad(pvscale, inhat, pvbias);
            ActivationFunction(1, &act_out, &bn_out, gamma, beta, alpha);
            out[index] = (_FLOAT)act_out;
        } // end for

        // Tail of loop
        inhat = (batchvalues[MIO_BN_NLOOPM] - (_FLOAT_PREC)mean) * ((_FLOAT_PREC)invVariance);
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        if(index < MIO_BN_NCHW)
        {
            bn_out = mad(pvscale, inhat, pvbias);
            ActivationFunction(1, &act_out, &bn_out, gamma, beta, alpha);
            out[index] = (_FLOAT)act_out;
        }
    }

    if(lid == 0)
    {
#if(MIO_RUNNING_RESULT == 1)
        running_stash(runningMean, runningVariance, expAvgFactor, mean, variance, grpid);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        saved_stash(savedMean, savedInvVariance, mean, invVariance, grpid);
#endif
    }
} // end spatial norm

#elif(MIO_BN_VARIANT == 1)

//===========

#if(MIO_BN_HW >= 4096)
#define MIO_MAX_READ 3
#else
#define MIO_MAX_READ 2
#endif
#define RD_BLK 1
#define GRPRD (MIO_BN_GRP0 * RD_BLK * 4)
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
MIOpenBatchNormActivFwdTrainSpatial(

    float INHW,
    const _FLOAT alpha,
    const _FLOAT beta,
    const _FLOAT gamma,
    double epsilon,
#if(MIO_RUNNING_RESULT == 1)
    double expAvgFactor,
#endif
    const __global _FLOAT* __restrict in,
    __global _FLOAT* __restrict out,
    __constant _FLOAT_PREC* __restrict bias,
    __constant _FLOAT_PREC* __restrict scale

#if(MIO_RUNNING_RESULT == 1)
    ,
    __global _FLOAT_PREC* __restrict runningMean,
    __global _FLOAT_PREC* __restrict runningVariance
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
    ,
    __global _FLOAT_PREC* __restrict savedInvVariance,
    __global _FLOAT_PREC* __restrict savedMean
#endif

)
{

    // SPATIAL

    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
    _FLOAT_PREC variance    = (_FLOAT_PREC)0.;
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pvscale, pvbias;
    _FLOAT_PREC bn_out, act_out;

    __local _FLOAT_PREC lcl_bias;
    __local _FLOAT_PREC lcl_scale;

    int index = 0;
    int lid   = get_local_id(0);
    int grpid = get_group_id(0);
    int chwid = grpid * MIO_BN_HW;
    int nidx  = 0;
    int hwidx = 0;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_HW >= 4096)
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
    if(index < MIO_BN_NCHW)
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
        index           = nidx * MIO_BN_CHW + chwid + hwidx;
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
        index               = nidx * MIO_BN_CHW + chwid + hwidx;
        _FLOAT_PREC xin     = (_FLOAT_PREC)((index < MIO_BN_NCHW) ? *(in + index) : 0.);
        mean += xin;
        variance = mad(xin, xin, variance);
    }
#endif
#endif
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

// REDUCE MEAN AND VARIANCE -----------------------
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
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + epsilon);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;

#if(MIO_BN_REM == 0)
    __attribute__((opencl_unroll_hint(2))) for(unsigned int k = lid; k < MIO_BN_LESS;
                                               k += MIO_BN_GRP0)
    {
        nidx   = k / MIO_BN_HW;
        hwidx  = k - (nidx * MIO_BN_HW);
        index  = nidx * MIO_BN_CHW + chwid + hwidx;
        bn_out = mad(pvscale, (*(in + index) - mean) * invVariance, pvbias);
        ActivationFunction(1, &act_out, &bn_out, gamma, beta, alpha);
        out[index] = (_FLOAT)act_out;

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
            bn_out         = mad(pvscale, xhat[j], pvbias);
            ActivationFunction(1, &act_out, &bn_out, gamma, beta, alpha);
            out[index] = (_FLOAT)act_out;
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
        _FLOAT_PREC xin = (index < MIO_BN_NCHW) ? ((_FLOAT_PREC)(*(in + index))) : 0.;
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
            bn_out = mad(pvscale, xhat[j], pvbias);
            ActivationFunction(1, &act_out, &bn_out, gamma, beta, alpha);
            out[index] = (_FLOAT)act_out;
        }
    }
#endif
#endif

    if(lid == 0)
    {
#if(MIO_RUNNING_RESULT == 1)
        running_stash(runningMean, runningVariance, expAvgFactor, mean, variance, grpid);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        saved_stash(savedMean, savedInvVariance, mean, invVariance, grpid);
#endif
    }

} // end spatial norm

#elif(MIO_BN_VARIANT == 2)
// MULTI-KERNEL reduction for > 33M elements

#elif(MIO_BN_VARIANT == 3)

// This kernel implies the image is greater than a wavefront, but smaller than 257
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormActivFwdTrainSpatial(

    float INHW,
    const _FLOAT alpha,
    const _FLOAT beta,
    const _FLOAT gamma,
    double epsilon,
#if(MIO_RUNNING_RESULT == 1)
    double expAvgFactor,
#endif
    const __global _FLOAT* __restrict in,
    __global _FLOAT* __restrict out,
    __constant _FLOAT_PREC* __restrict bias,
    __constant _FLOAT_PREC* __restrict scale

#if(MIO_RUNNING_RESULT == 1)
    ,
    __global _FLOAT_PREC* __restrict runningMean,
    __global _FLOAT_PREC* __restrict runningVariance
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
    ,
    __global _FLOAT_PREC* __restrict savedInvVariance,
    __global _FLOAT_PREC* __restrict savedMean
#endif

)
{
    // SPATIAL
    _FLOAT_PREC mean        = (_FLOAT)0.;
    _FLOAT_PREC variance    = (_FLOAT)0.;
    _FLOAT_PREC invVariance = (_FLOAT)0.;
    _FLOAT_PREC inhat       = (_FLOAT)0.;
    _FLOAT_PREC pvscale, pvbias;
    _FLOAT_PREC bn_out, act_out;

    __local _FLOAT_PREC lcl_bias;
    __local _FLOAT_PREC lcl_scale;

    unsigned int index;
    unsigned int lid   = get_local_id(0);
    unsigned int grpid = get_group_id(0);
    unsigned int cidx  = grpid * MIO_BN_HW;

#if(MIO_BN_N < MIO_BN_MAXN)
    _FLOAT_PREC minibatch[MIO_BN_N];
#endif

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // MEAN
    if(lid < MIO_BN_HW)
    {
        __attribute__((opencl_unroll_hint(2))) for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index        = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            minibatch[n] = (_FLOAT_PREC)(*(in + index));
            mean += minibatch[n];
            variance = mad(minibatch[n], minibatch[n], variance);
#else
            _FLOAT_PREC xin = (_FLOAT_PREC)(*(in + index));
            mean += xin;
            variance = mad(xin, xin, variance);
#endif
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

#if !MIOPEN_USE_AMDGCN
#if MIOPEN_USE_FP16 == 1
    local float lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[lid] = (float)mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_mean = (float)mean;
    regLDSreduce(&temp_mean, lcl_data, lid, (float)INHW);
    mean = (_FLOAT_PREC)temp_mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = (float)variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_variance = (float)variance;
    regLDSreduce(&temp_variance, lcl_data, lid, (float)INHW);
    variance = (_FLOAT_PREC)temp_variance;
#else
    __local _FLOAT_PREC lcl_data[MIO_BN_LDS_SIZE];

    // Reduce mean
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, (_FLOAT_PREC)INHW);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce variance
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, (_FLOAT_PREC)INHW);

#endif
#else
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x2, lcl_data_y2, lid);

#endif

    barrier(CLK_LOCAL_MEM_FENCE);
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + (_FLOAT_PREC)epsilon);

    // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if(lid < MIO_BN_HW)
    {
        pvscale = lcl_scale;
        pvbias  = lcl_bias;

        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index  = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            inhat  = (minibatch[n] - mean) * invVariance; // (in[index] - mean) * invVariance;
#else
            inhat = ((_FLOAT_PREC)(*(in + index)) - mean) * invVariance;
#endif
            bn_out = mad(pvscale, inhat, pvbias);
            ActivationFunction(1, &act_out, &bn_out, gamma, beta, alpha);
            out[index] = (_FLOAT)act_out;

        } // end for
    }     // end if

    if(lid == 0)
    {
#if(MIO_RUNNING_RESULT == 1)
        running_stash(runningMean, runningVariance, expAvgFactor, mean, variance, grpid);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        saved_stash(savedMean, savedInvVariance, mean, invVariance, grpid);
#endif
    }

} // end spatial norm

#endif

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
