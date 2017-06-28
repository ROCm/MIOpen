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
#define MIO_BN_LDS_SIZE 256
#endif

#ifndef MIO_BN_LDS_NSIZE
#define MIO_BN_LDS_NSIZE 256
#endif

#ifndef MIO_BN_LDS_HWSIZE
#define MIO_BN_LDS_HWSIZE 256
#endif

#ifndef MIO_BN_C
#define MIO_BN_C 1
#endif

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
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
#define MIO_BN_VARIANT 4
#endif

#ifdef __AMDGCN__
#undef __AMDGCN__
#endif

#define UNUSED __attribute__((__unused__))

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
BatchNormFwdTrainSpatial(const __global _FLOAT* __restrict in,
                         __global _FLOAT* __restrict out,
                         __constant _FLOAT* __restrict scale,
                         __constant _FLOAT* __restrict bias,
                         float INHW,
#if(MIO_RUNNING_RESULT == 1)
                         double expAvgFactor,
                         __global _FLOAT* __restrict resultRunningMean,
                         __global _FLOAT* __restrict resultRunningVariance,
#endif
                         double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                         ,
                         __global _FLOAT* __restrict resultSaveMean,
                         __global _FLOAT* __restrict resultSaveInvVariance
#endif
                         )
{

    // SPATIAL
    _FLOAT mean        = 0.;
    _FLOAT variance    = 0.;
    _FLOAT invVariance = 0.;
    _FLOAT inhat       = 0.;
    _FLOAT elemStd     = 0.;
    _FLOAT pvscale, pvbias;

    __local _FLOAT lcl_indata[MIO_BN_LDS_HWSIZE][MIO_BN_LDS_NSIZE];
    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;

    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid * MIO_BN_HW;

    if(ylid == 0)
    {
        lcl_scale = scale[xgid];
        lcl_bias  = bias[xgid];
    }

    if(ygid < MIO_BN_N)
    {
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index                        = ygid * MIO_BN_CHW + cidx + hw;
            mean += lcl_indata[hw][ylid] = in[index];
        }
    }

#ifdef __AMDGCN__
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
    mean *= INHW;
#else

    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0] * INHW;
#endif

    if(ygid < MIO_BN_N)
    {
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            elemStd  = (lcl_indata[hw][ylid] - mean);
            variance = mad(elemStd, elemStd, variance);
        }
    }

#ifdef __AMDGCN__

    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance = as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    variance *= INHW;
#else

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
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

#endif

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_N)
    {
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index      = ygid * MIO_BN_CHW + cidx + hw;
            inhat      = (lcl_indata[hw][ylid] - mean) * invVariance;
            out[index] = mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0)
    {
// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[xgid]        = mean;
        resultSaveInvVariance[xgid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean = resultRunningMean[xgid];
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[xgid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        _FLOAT rtmp                 = mad((_FLOAT)-expAvgFactor, adjust, adjust);
        resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor, resultRunningVariance[xgid], rtmp);
#endif
    }
#endif
} // end spatial norm

#elif(MIO_BN_VARIANT == 1)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdTrainSpatial(const __global _FLOAT* __restrict in,
                         __global _FLOAT* __restrict out,
                         __constant _FLOAT* __restrict scale,
                         __constant _FLOAT* __restrict bias,
                         float INHW,
#if(MIO_RUNNING_RESULT == 1)
                         double expAvgFactor,
                         __global _FLOAT* __restrict resultRunningMean,
                         __global _FLOAT* __restrict resultRunningVariance,
#endif
                         double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                         ,
                         __global _FLOAT* __restrict resultSaveMean,
                         __global _FLOAT* __restrict resultSaveInvVariance
#endif
                         )
{

    // SPATIAL
    _FLOAT mean        = 0.;
    _FLOAT variance    = 0.;
    _FLOAT invVariance = 0.;
    _FLOAT inhat       = 0.;
    _FLOAT elemStd     = 0.;
    _FLOAT pvscale, pvbias;

    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;

    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid * MIO_BN_HW;
    unsigned int idx  = cidx + ygid;

    if(ylid == 0)
    {
        lcl_scale = scale[xgid];
        lcl_bias  = bias[xgid];
    }

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index                       = n * MIO_BN_CHW + idx;
            mean += lcl_indata[n][ylid] = in[index];
        }
    }

#ifdef __AMDGCN__

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
    mean *= INHW;

#else

    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0] * INHW;

#endif

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            elemStd  = (lcl_indata[n][ylid] - mean);
            variance = mad(elemStd, elemStd, variance);
        }
    }

#ifdef __AMDGCN__

    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance = as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    variance *= INHW;

#else

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
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

#endif

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    pvscale = lcl_scale;
    pvbias  = lcl_bias;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index      = n * MIO_BN_CHW + idx;
            inhat      = (lcl_indata[n][ylid] - mean) * invVariance;
            out[index] = mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0)
    {
// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[xgid]        = mean;
        resultSaveInvVariance[xgid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean          = resultRunningMean[xgid];
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[xgid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        _FLOAT rtmp                 = mad((_FLOAT)-expAvgFactor, adjust, adjust);
        resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor, resultRunningVariance[xgid], rtmp);
#endif
    }
#endif
} // end spatial norm

#elif(MIO_BN_VARIANT == 2)

// This kernel implied that the input data does not fit into LDS, but the image size is
// smaller than 64 pixels

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdTrainSpatial(const __global _FLOAT* __restrict in,
                         __global _FLOAT* __restrict out,
                         __constant _FLOAT* __restrict scale,
                         __constant _FLOAT* __restrict bias,
                         float INHW,
#if(MIO_RUNNING_RESULT == 1)
                         double expAvgFactor,
                         __global _FLOAT* __restrict resultRunningMean,
                         __global _FLOAT* __restrict resultRunningVariance,
#endif
                         double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                         ,
                         __global _FLOAT* __restrict resultSaveMean,
                         __global _FLOAT* __restrict resultSaveInvVariance
#endif
                         )
{

    // SPATIAL
    _FLOAT mean     = 0.;
    _FLOAT variance = 0.;
    _FLOAT invVariance, inhat, elemStd;
    _FLOAT pvscale, pvbias;

    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;

    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid * MIO_BN_HW;
    unsigned int idx  = cidx + ygid;

    if(ylid == 0)
    {
        lcl_scale = scale[xgid];
        lcl_bias  = bias[xgid];
    }

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + idx;
            mean += in[index];
        }
    }

#ifdef __AMDGCN__
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
    mean *= INHW;

#else

    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0] * INHW;
#endif

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index    = n * MIO_BN_CHW + idx;
            elemStd  = (in[index] - mean);
            variance = mad(elemStd, elemStd, variance);
        }
    }

#ifdef __AMDGCN__

    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance = as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    variance *= INHW;

#else

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
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

#endif

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    if(ygid < MIO_BN_HW)
    {

        pvscale = lcl_scale;
        pvbias  = lcl_bias;

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index = n * MIO_BN_CHW + idx;
            // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            inhat = (in[index] - mean) * invVariance;
            // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
            out[index] = mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0)
    {

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[xgid]        = mean;
        resultSaveInvVariance[xgid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean          = resultRunningMean[xgid];
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[xgid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        _FLOAT rtmp                 = mad((_FLOAT)-expAvgFactor, adjust, adjust);
        resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor, resultRunningVariance[xgid], rtmp);
#endif
    }
#endif
} // end spatial norm

#elif(MIO_BN_VARIANT == 3)

// This kernel implies the image is greater than a wavefront, but smaller than 257

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdTrainSpatial(const __global _FLOAT* __restrict in,
                         __global _FLOAT* __restrict out,
                         __constant _FLOAT* __restrict scale,
                         __constant _FLOAT* __restrict bias,
                         float INHW,
#if(MIO_RUNNING_RESULT == 1)
                         double expAvgFactor,
                         __global _FLOAT* __restrict resultRunningMean,
                         __global _FLOAT* __restrict resultRunningVariance,
#endif
                         double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                         ,
                         __global _FLOAT* __restrict resultSaveMean,
                         __global _FLOAT* __restrict resultSaveInvVariance
#endif
                         )
{

    // SPATIAL

    _FLOAT mean     = 0.;
    _FLOAT variance = 0.;
    _FLOAT invVariance, inhat, elemStd;
    _FLOAT pvscale, pvbias;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT2 lcl_scalebias;

    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid * MIO_BN_HW;
    unsigned int idx  = cidx + ygid;

    if(ylid == 0)
    {
        lcl_scalebias[0] = scale[xgid];
        lcl_scalebias[1] = bias[xgid];
    }

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + idx;
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
    mean = as_float(__builtin_amdgcn_readlane(as_int(mean), 63));
    if(ylid == 0)
        lcl_data[0] = mean * INHW;
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0];

#else

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0] * INHW;

#endif

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index    = n * MIO_BN_CHW + idx;
            elemStd  = (in[index] - mean);
            variance = mad(elemStd, elemStd, variance);
        }
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__

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
    variance = as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    if(ylid == 0)
        lcl_data[0] = variance * INHW;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0];

#else

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

#endif

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);
    // DONE WITH variance

    if(ygid < MIO_BN_HW)
    {

        pvscale = lcl_scalebias[0];
        pvbias  = lcl_scalebias[1];

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index = n * MIO_BN_CHW + idx;
            // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            inhat = (in[index] - mean) * invVariance;
            // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
            out[index] = mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0)
    {

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[xgid]        = mean;
        resultSaveInvVariance[xgid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean          = resultRunningMean[xgid];
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[xgid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        _FLOAT rtmp                 = mad((_FLOAT)-expAvgFactor, adjust, adjust);
        resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor, resultRunningVariance[xgid], rtmp);
#endif
    }
#endif
} // end spatial norm

#elif(MIO_BN_VARIANT == 4)

// This kernel implies the image is greater than a wavefront, but smaller than 257

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdTrainSpatial(const __global _FLOAT* __restrict in,
                         __global _FLOAT* __restrict out,
                         __constant _FLOAT* __restrict scale,
                         __constant _FLOAT* __restrict bias,
                         float INHW,
#if(MIO_RUNNING_RESULT == 1)
                         double expAvgFactor,
                         __global _FLOAT* __restrict resultRunningMean,
                         __global _FLOAT* __restrict resultRunningVariance,
#endif
                         double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                         ,
                         __global _FLOAT* __restrict resultSaveMean,
                         __global _FLOAT* __restrict resultSaveInvVariance
#endif
                         )
{

    // SPATIAL

    _FLOAT mean     = 0.;
    _FLOAT variance = 0.;
    _FLOAT invVariance, inhat, elemStd;
    _FLOAT pvscale, pvbias;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;

    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int cidx = xgid * MIO_BN_HW;
    unsigned int idx  = cidx + ygid;

    if(ylid == 0)
    {
        lcl_scale = scale[xgid];
        lcl_bias  = bias[xgid];
    }

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index                       = n * MIO_BN_CHW + idx;
            mean += lcl_indata[n][ylid] = in[index];
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
    mean = as_float(__builtin_amdgcn_readlane(as_int(mean), 63));
    if(ylid == 0)
        lcl_data[0] = mean * INHW;
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0];

#else

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lcl_data[0] * INHW;

#endif

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index   = n * MIO_BN_CHW + idx;
            elemStd = lcl_indata[n][ylid] = (lcl_indata[n][ylid] - mean);
            variance                      = mad(elemStd, elemStd, variance);
        }
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__

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
    variance = as_float(__builtin_amdgcn_readlane(as_int(variance), 63));
    if(ylid == 0)
        lcl_data[0] = variance * INHW;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lcl_data[0];

#else

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

#endif

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);
    // DONE WITH variance

    if(ygid < MIO_BN_HW)
    {

        pvscale = lcl_scale;
        pvbias  = lcl_bias;

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index = n * MIO_BN_CHW + idx;
            // #4 apply the normalization, x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            inhat = (in[index] - mean) * invVariance;
            // #5 Gamma and Beta adjust, y_i = gamma*x_hat + beta
            // out[index] = mad(pvt_scale, inhat, pvt_bias);
            out[index] = mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ygid == 0)
    {

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[xgid]        = mean;
        resultSaveInvVariance[xgid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean          = resultRunningMean[xgid];
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[xgid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        _FLOAT rtmp                 = mad((_FLOAT)-expAvgFactor, adjust, adjust);
        resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor, resultRunningVariance[xgid], rtmp);
#endif
    }
#endif
} // end spatial norm

#elif(MIO_BN_VARIANT == 5)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdTrainSpatialNorm(const __global _FLOAT* __restrict in,
                             __global _FLOAT* __restrict out,
                             const __global _FLOAT* __restrict scale,
                             const __global _FLOAT* __restrict bias)
{

    // SPATIAL
    _FLOAT mean        = 0.;
    _FLOAT invVariance = 0.;
    _FLOAT inhat       = 0.;
    _FLOAT pvt_scale   = 0.;
    _FLOAT pvt_bias    = 0.;

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
        lcl_scale = scale[xgid];
        lcl_bias  = bias[xgid];
        lcl_mean  = out[meanstashindex]; // load stashed mean
        lcl_ivar  = out[varstashindex];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
        mean        = lcl_mean;
        invVariance = lcl_ivar;
        pvt_scale   = lcl_scale;
        pvt_bias    = lcl_bias;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index = n * MIO_BN_CHW + cidx + ygid;
            inhat = (in[index] - mean) * invVariance;
            // #5 Gamma and Beta adjust :: y_i = gamma*x_hat + beta
            out[index] = mad(pvt_scale, inhat, pvt_bias);
        } // end for(n)
    }     // end if(inImgIndex)
} // end spatial norm

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdTrainSpatialFinalVariance(__global _FLOAT* __restrict varbuff,
                                      float INHW
#if(MIO_RUNNING_RESULT == 1)
                                      ,
                                      double expAvgFactor,
                                      __global _FLOAT* __restrict resultRunningVariance
#endif
                                      ,
                                      double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                                      ,
                                      __global _FLOAT* __restrict resultSaveInvVariance
#endif
                                      )
{

    // SPATIAL
    __private _FLOAT variance    = 0.;
    __private _FLOAT invVariance = 0.;

    unsigned int ylid          = get_local_id(1);
    unsigned int ygrp_id       = get_group_id(1);
    unsigned int xgid          = get_global_id(0);
    unsigned int ygrp_sz       = get_local_size(1);
    unsigned int yngrps        = get_num_groups(1);
    unsigned int cidx          = xgid * MIO_BN_HW;
    unsigned int varstashindex = cidx + ygrp_sz * ygrp_id + 3;
    unsigned int commitID      = 0;

    for(int gn = 0; gn < yngrps; gn++)
    {
        unsigned int offset   = gn * ygrp_sz + ylid;
        unsigned int varindex = cidx + ygrp_sz * offset + 2;
        if(offset < yngrps)
        {                                  // modified to span larger number of groups
            variance += varbuff[varindex]; // load per group variance
        }
    }

#if(MIO_BN_NGRPS > 64)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__

    _FLOAT tmp = 0.;

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
    variance *= INHW;
    commitID = 63;

#else

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
    commitID = 0;

#endif

#elif(MIO_BN_NGRPS > 16)

#ifdef __AMDGCN__

    _FLOAT tmp = 0.;
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x111, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x112, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x114, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x142, 15, 15, 0));
    variance += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x143, 15, 15, 0));
    variance += tmp;
    variance *= INHW;
    commitID = 63;
#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
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
    commitID = 0;

#endif

#else //(MIO_BN_NGRPS <= 16)

#ifdef __AMDGCN__

    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x108, 15, 15, 0));

#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        variance += lcl_data[i];
    }

#endif
    variance *= INHW;
    commitID = 0;

#endif

    invVariance = rsqrt(variance + epsilon);
    if(ylid == commitID)
    {
        varbuff[varstashindex] = invVariance; // stash mean
    }
#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(get_global_id(1) == commitID)
    {
// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveInvVariance[xgid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        // var(n+1) = p * var(n-1) + (1 - p)*(b/b-1)*var(n)
        // right:: (1 - p)*(b/b-1)*var(n) = (1 - p)*adjust = -p*adjust + adjust
        // var(n+1) = (p* var(n-1)) +  (-p*adjust + adjust)
        _FLOAT NHW                  = (_FLOAT)MIO_BN_NHW;
        const _FLOAT adjust         = (MIO_BN_NHW == 1) ? variance : variance * (NHW / (NHW - 1));
        const _FLOAT rtmp           = mad((_FLOAT)-expAvgFactor, adjust, adjust);
        resultRunningVariance[xgid] = mad((_FLOAT)expAvgFactor, resultRunningVariance[xgid], rtmp);
#endif
    }
#endif
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdTrainSpatialVariance(const __global _FLOAT* __restrict in, /* x input */
                                 __global _FLOAT* __restrict meanvarbuff)
{

    // SPATIAL
    _FLOAT variance = 0.;
    _FLOAT mean, elemStd;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_mean;
    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int cidx    = xgid * MIO_BN_HW;
    unsigned int index;

    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;

    if(ylid == 0)
    {
        lcl_mean = meanvarbuff[meanstashindex]; // load stashed mean
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
        mean = lcl_mean;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index   = n * MIO_BN_CHW + cidx + ygid;
            elemStd = (in[index] - mean);
            variance += elemStd * elemStd;
        }
    }

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__

    _FLOAT tmp = 0.;
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
        unsigned int varindex = cidx + ygrp_sz * ygrp_id + 2;
        meanvarbuff[varindex] = variance; // pre-stage for group reduction
    }

#else

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
        unsigned int varindex = cidx + ygrp_sz * ygrp_id + 2;
        meanvarbuff[varindex] = lcl_data[0]; // pre-stage for group reduction
    }
#endif

} // end spatial variance

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdTrainSpatialFinalMean(__global _FLOAT* __restrict meanvarbuff,
                                  float INHW
#if(MIO_RUNNING_RESULT == 1)
                                  ,
                                  double expAvgFactor /* input momentum */
                                  ,
                                  __global _FLOAT* __restrict resultRunningMean /*input and output*/
#endif
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                                  ,
                                  __global _FLOAT* __restrict resultSaveMean /*output only*/
#endif
                                  )
{

    _FLOAT mean           = 0.;
    unsigned int ylid     = get_local_id(1);
    unsigned int ygrp_id  = get_group_id(1);
    unsigned int xgid     = get_global_id(0);
    unsigned int ygrp_sz  = get_local_size(1);
    unsigned int yngrps   = get_num_groups(1);
    unsigned int cidx     = xgid * MIO_BN_HW;
    unsigned int commitID = 0;

    for(int gn = 0; gn < yngrps; gn++)
    {
        unsigned int offset    = gn * ygrp_sz + ylid;
        unsigned int meanindex = cidx + ygrp_sz * offset;
        if(offset < yngrps)
        { // modify to span larger number of groups
            mean += meanvarbuff[meanindex];
        }
    }

#if(MIO_BN_NGRPS > 64)

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
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
    mean *= INHW;
    commitID = 63;

#else

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean     = lcl_data[0] * INHW;
    commitID = 0;

#endif

#elif(MIO_BN_NGRPS > 16)

#ifdef __AMDGCN__

    _FLOAT tmp = 0.;
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x111, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x112, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x114, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x142, 15, 15, 0));
    mean += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x143, 15, 15, 0));
    mean += tmp;
    mean *= INHW;
    commitID = 63;

#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(lcl_data, 1, ylid, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(lcl_data, 4, ylid, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    mean     = lcl_data[0] * INHW;
    commitID = 0;

#endif

#else

#ifdef __AMDGCN__

    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x108, 15, 15, 0));

#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        mean += lcl_data[i];
    }

#endif

    mean *= INHW;
    commitID = 0;

#endif

    if(ylid == commitID)
    {
        unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
        meanvarbuff[meanstashindex] = mean; // stash mean
    }

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    unsigned int ygid = get_global_id(1);
    if(ygid == commitID)
    {
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[xgid] = mean;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean   = resultRunningMean[xgid];
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[xgid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
#endif
    }
#endif
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormFwdTrainSpatialMean(const __global _FLOAT* __restrict in,
                             __global _FLOAT* __restrict meanbuff)
{

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx      = xgid * MIO_BN_HW;
    unsigned int meanindex = cidx + ygrp_sz * ygrp_id;
    _FLOAT mean            = 0.;

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
        meanbuff[meanindex] = mean;
    }

#else

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
        meanbuff[meanindex] = lcl_data[0];
    }
#endif

} // end spatial mean kernel

#endif

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
