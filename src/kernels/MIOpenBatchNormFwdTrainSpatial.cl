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

#ifndef MIO_BN_NLOOP
#define MIO_BN_NLOOP 1
#endif

#ifndef MIO_BN_NCHW
#define MIO_BN_NCHW 1
#endif

#ifndef MIO_BN_VARIANT
#define MIO_BN_VARIANT 3
#endif

#ifndef MIO_BN_SEGMENT
#define MIO_BN_SEGMENT 1
#endif

#ifndef MIO_BN_SEGIHW
#define MIO_BN_SEGIHW 1
#endif

#define MIO_BN_MAXN 512

/*
#ifdef __AMDGCN__
#undef __AMDGCN__
#endif
*/

#define UNUSED __attribute__((__unused__))

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

__attribute__((always_inline)) uint iDiv(uint v, uint d)
{
    uint r = (uint)((float)v * (1.f / (float)d) + 0.00001f);
    return (r);
}

__attribute__((always_inline)) uint iMod(uint v, uint u, uint d)
{
    uint r = v - mul24(u, d);
    return (r);
}

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

static inline void
regLDSreduce(_FLOAT* value, __local _FLOAT* data, unsigned int localID, _FLOAT scale)
{
    data[localID] = *value;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (MIO_BN_LDS_SIZE >> 2))
        ReduceKernel(data, 1, localID, 4);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID < (MIO_BN_LDS_SIZE >> 4))
        ReduceKernel(data, 4, localID, 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(localID == 0)
        ReduceKernel(data, 16, localID, MIO_BN_LDS_SIZE);
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0] * scale;
}

#ifdef __AMDGCN__
static inline void dppRegReduce64(_FLOAT* value, _FLOAT scale)
{
    _FLOAT tmp = 0.;
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x111, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x112, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x114, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x118, 0xF, 0xF, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x142, 0xF, 0xF, 0));
    *value += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x143, 0xF, 0xF, 0));
    *value += tmp;
    *value = as_float(__builtin_amdgcn_readlane(as_int(*value), 63));
    *value *= scale;
}

static inline void dppRegReduce16(_FLOAT* value, _FLOAT scale)
{

    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x101, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x102, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x104, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x108, 0xF, 0xF, 0));
    *value = as_float(__builtin_amdgcn_readlane(as_int(*value), 0));
    *value *= scale;
}

static inline void
dppLDSReduce64(_FLOAT* value, __local _FLOAT* data, unsigned int localID, _FLOAT scale)
{
    _FLOAT tmp = 0.;
    *value     = data[localID];
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x111, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x112, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x114, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x118, 0xF, 0xF, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x142, 0xF, 0xF, 0));
    *value += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x143, 0xF, 0xF, 0));
    *value += tmp;
    if(localID == 63)
        data[0] = *value * scale;
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0];
}

static inline void
dppLDSReduce16(_FLOAT* value, __local _FLOAT* data, unsigned int localID, _FLOAT scale)
{

    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x101, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x102, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x104, 0xF, 0xF, 0));
    *value += as_float(__builtin_amdgcn_mov_dpp(as_int(*value), 0x108, 0xF, 0xF, 0));
    if(localID == 0)
        data[0] = *value * scale;
    barrier(CLK_LOCAL_MEM_FENCE);
    *value = data[0];
}
#endif

#if(MIO_BN_VARIANT == 255)

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
    _FLOAT pvscale     = elemStd;
    _FLOAT pvbias      = 0.;

    _FLOAT batchvalues[MIO_BN_NLOOP];

    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int index  = 0;
    unsigned int lid    = get_local_id(0);
    unsigned int grpid  = get_group_id(0);
    unsigned int chwid  = grpid * MIO_BN_HW + (lid % MIO_BN_HW);
    unsigned int lidihw = lid / MIO_BN_HW;
    unsigned int segihw = MIO_BN_SEGIHW;
    unsigned int nid    = 0;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }

    if(lid < MIO_BN_SEGMENT)
    {
//==== CALC MEAN =======================
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_NLOOP; n++)
        {
            nid            = n * segihw + lidihw;
            index          = nid * MIO_BN_CHW + chwid;
            batchvalues[n] = (index < MIO_BN_NCHW) ? *(in + index) : 0.;
            mean += batchvalues[n]; // = 1.;//(index < MIO_BN_NCHW) ? in[index] : 0.;
            // printf("mean: %f\n",mean);
        }
    }
    // barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&mean, lcl_data, lid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, INHW);
#endif

    if(lid < MIO_BN_SEGMENT)
    {
//==== CALC VARIANCE =======================
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_NLOOP; n++)
        {
            nid            = n * segihw + lidihw;
            batchvalues[n] = (batchvalues[n] - mean);
            if(nid < MIO_BN_N)
            {
                variance = mad(batchvalues[n], batchvalues[n], variance);
            }
            else
            {
                variance = 0.;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&variance, lcl_data, lid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, INHW);
#endif

    invVariance = rsqrt(variance + epsilon);

    if(lid < MIO_BN_SEGMENT)
    {

        //==== CALC NORM =======================
        pvscale = lcl_scale;
        pvbias  = lcl_bias;

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_NLOOP; n++)
        { // apply normalization
            // inhat = (batchvalues[n] - mean) * invVariance;
            inhat = batchvalues[n] * invVariance;
            nid   = n * segihw + lidihw;

            index = nid * MIO_BN_CHW + chwid;
            if(index < MIO_BN_NCHW)
                out[index] = mad(pvscale, inhat, pvbias);
        } // end for
    }
#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(lid == 0)
    {

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        *(resultSaveMean + grpid)        = mean;
        *(resultSaveInvVariance + grpid) = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean = *(resultRunningMean + grpid);
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[grpid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        resultRunningVariance[grpid] =
            (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + grpid) +
            (_FLOAT)expAvgFactor * adjust;
#endif
    }
#endif
} // end spatial norm

#elif(MIO_BN_VARIANT == 0)

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
    _FLOAT pvscale, pvbias;
    _FLOAT minibatch[MIO_BN_HW];
    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;

#ifndef __AMDGCN__
#if(MIO_BN_N > 1)
    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
#endif
#endif

    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int cidx = xgid * MIO_BN_HW;

    if(ylid == 0)
    {
        lcl_scale = *(scale + xgid);
        lcl_bias  = *(bias + xgid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < MIO_BN_N)
    {
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index                 = ylid * MIO_BN_CHW + cidx + hw;
            mean += minibatch[hw] = *(in + index);
        }
    }
    else
    {
        mean = 0.;
    }

#ifdef __AMDGCN__

#if(MIO_BN_N > 16)
    dppRegReduce64(&mean, INHW);
#elif(MIO_BN_N > 1) // N
    dppRegReduce16(&mean, INHW);
#else
    mean *= INHW;
#endif // N

#else // GCN

#if(MIO_BN_N > 16)
    regLDSreduce(&mean, lcl_data, ylid, INHW);
#elif(MIO_BN_N > 1)
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_N; i++)
    {
        mean += lcl_data[i];
    }
    mean *= INHW;
#else
    mean *= INHW;
#endif // N

#endif // GCN

    if(ylid < MIO_BN_N)
    {
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            minibatch[hw] = minibatch[hw] - mean;
            variance      = mad(minibatch[hw], minibatch[hw], variance);
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_N > 16)
    dppRegReduce64(&variance, INHW);
#elif(MIO_BN_N > 1)
    dppRegReduce16(&variance, INHW);
#else
    variance *= INHW;
#endif // N

#else // GCN

#if(MIO_BN_N > 16)
    regLDSreduce(&variance, lcl_data, ylid, INHW);
#elif(MIO_BN_N > 1)
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_N; i++)
    {
        variance += lcl_data[i];
    }
    variance *= INHW;
#else
    variance *= INHW;
#endif // N
#endif // GCN

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    if(ylid < MIO_BN_N)
    {
        pvscale = lcl_scale;
        pvbias  = lcl_bias;

#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index      = ylid * MIO_BN_CHW + cidx + hw;
            inhat      = minibatch[hw] * invVariance;
            out[index] = mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(get_global_id(1) == 0)
    {
// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[xgid]        = mean;
        resultSaveInvVariance[xgid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean          = *(resultRunningMean + xgid);
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[xgid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        resultRunningVariance[xgid] = (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + xgid) +
                                      (_FLOAT)expAvgFactor * adjust;
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
    _FLOAT elemStd     = 0.;
    _FLOAT inhat       = elemStd;
    _FLOAT pvscale, pvbias;
    _FLOAT minibatch[MIO_BN_N];

    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;

#ifndef __AMDGCN__
#if(MIO_BN_HW > 1)
    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
#endif
#endif

    unsigned int index;
    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int idx  = xgid * MIO_BN_HW + ylid;

    if(ylid == 0)
    {
        lcl_scale = *(scale + xgid);
        lcl_bias  = *(bias + xgid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index                = n * MIO_BN_CHW + idx;
            mean += minibatch[n] = *(in + index);
        }
    }
    else
    {
        mean = 0.;
    }

#ifdef __AMDGCN__

#if(MIO_BN_HW > 16)
    dppRegReduce64(&mean, INHW);
#elif(MIO_BN_HW > 1)
    dppRegReduce16(&mean, INHW);
#else
    mean *= INHW;
#endif // HW

#else // GCN

#if(MIO_BN_HW > 16)
    regLDSreduce(&mean, lcl_data, ylid, INHW);
#elif(MIO_BN_HW > 1)
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = 0.;
    for(int i = 0; i < MIO_BN_HW; i++)
    {
        mean += lcl_data[i];
    }
    mean *= INHW;
#else
    mean *= INHW;
#endif // HW
#endif // GCN

    if(ylid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            minibatch[n] = minibatch[n] - mean;
            variance     = mad(minibatch[n], minibatch[n], variance);
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_HW > 16)
    dppRegReduce64(&variance, INHW);
#elif(MIO_BN_HW > 1)
    dppRegReduce16(&variance, INHW);
#else
    variance *= INHW;
#endif // HW

#else // if not GCN

#if(MIO_BN_HW > 16)
    regLDSreduce(&variance, lcl_data, ylid, INHW);
#elif(MIO_BN_HW > 1)
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_HW; i++)
    {
        variance += lcl_data[i];
    }
    variance *= INHW;
#else
    variance *= INHW;
#endif // HW
#endif // GCN

    // #3 add epsilon for numeric stability, sq_root, and invert
    invVariance = rsqrt(variance + epsilon);

    if(ylid < MIO_BN_HW)
    {

        pvscale = lcl_scale;
        pvbias  = lcl_bias;

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index      = n * MIO_BN_CHW + idx;
            inhat      = minibatch[n] * invVariance;
            out[index] = mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(get_global_id(1) == 0)
    {
// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[xgid]        = mean;
        resultSaveInvVariance[xgid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean          = *(resultRunningMean + xgid);
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[xgid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        resultRunningVariance[xgid] = (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + xgid) +
                                      (_FLOAT)expAvgFactor * adjust;
#endif
    }
#endif
} // end spatial norm

#elif(MIO_BN_VARIANT == 2)

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

    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;

    unsigned int index;
    unsigned int ylid    = get_local_id(1);
    unsigned int xgid    = get_global_id(0);

#ifdef __AMDGCN__
    unsigned int segment = MIO_BN_GRP1 >> 6;
#endif

#if(MIO_BN_N < MIO_BN_MAXN)
    _FLOAT minibatch[MIO_BN_N];
#endif

    if(ylid == 0)
    {
        lcl_scale = *(scale + xgid);
        lcl_bias  = *(bias + xgid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // MEAN
    if(ylid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + xgid * MIO_BN_HW + ylid;
#if(MIO_BN_N < MIO_BN_MAXN)
            mean += minibatch[n] = *(in + index);
#else
            mean += *(in + index);
#endif
        }
    }

    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef __AMDGCN__
    mean = 0.;
    if(ylid < 64)
    {
        for(unsigned int red = 0; red < segment; red++)
        {
            mean += lcl_data[ylid * segment + red];
        }
    }
    else
    {
        mean = 0.;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    dppLDSReduce64(&mean, lcl_data, ylid, INHW);

#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, ylid, INHW);
#endif

    // VARIANCE
    if(ylid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {

#if(MIO_BN_N < MIO_BN_MAXN)
            elemStd = minibatch[n] = minibatch[n] - mean; //(in[index] - mean);
#else
            index   = n * MIO_BN_CHW + xgid * MIO_BN_HW + ylid;
            elemStd = (*(in + index) - mean);
#endif
            variance               = mad(elemStd, elemStd, variance);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
    variance = 0.;
    if(ylid < 64)
    {
        for(unsigned int red = 0; red < segment; red++)
        {
            variance += lcl_data[ylid * segment + red];
        }
    }
    else
    {
        variance = 0.;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    dppLDSReduce64(&variance, lcl_data, ylid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, ylid, INHW);
#endif

    invVariance = rsqrt(variance + epsilon);

    if(ylid < MIO_BN_HW)
    {
        pvscale = lcl_scale;
        pvbias  = lcl_bias;

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index      = n * MIO_BN_CHW + xgid * MIO_BN_HW + ylid;
#if(MIO_BN_N < MIO_BN_MAXN)
            inhat      = minibatch[n] * invVariance; // (in[index] - mean) * invVariance;
#else
            inhat = (*(in + index) - mean) * invVariance;
#endif
            out[index] = mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(get_global_id(1) == 0)
    {

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        *(resultSaveMean + xgid)        = mean;
        *(resultSaveInvVariance + xgid) = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean              = *(resultRunningMean + xgid);
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[xgid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        resultRunningVariance[xgid] = (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + xgid) +
                                      (_FLOAT)expAvgFactor * adjust;
#endif
    }
#endif
} // end spatial norm

#elif(MIO_BN_VARIANT == 3)

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
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index = n * MIO_BN_CHW + cidx + ygid;
            inhat = (*(in + index) - mean) * invVariance;
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
        {                                      // modified to span larger number of groups
            variance += *(varbuff + varindex); // load per group variance
        }
    }

#if(MIO_BN_NGRPS > 256)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&variance, lcl_data, ylid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, ylid, INHW);
#endif
#elif(MIO_BN_NGRPS > 64)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
    commitID = 63;
#pragma unroll
    for(unsigned int red = 128; red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&variance, lcl_data, ylid, INHW);
#else
    regLDSreduce(&variance, lcl_data, ylid, INHW);
#endif

#elif(MIO_BN_NGRPS > 16)

#ifdef __AMDGCN__
    commitID = 63;
    dppRegReduce64(&variance, INHW);
#else
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    commitID = 0;
    regLDSreduce(&variance, lcl_data, ylid, INHW);
#endif

#else //(MIO_BN_NGRPS <= 16)
    commitID = 0;

#ifdef __AMDGCN__
    dppRegReduce16(&variance, INHW);
#else
    __local _FLOAT lcl_data[16];
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        variance += lcl_data[i];
    }
    variance *= INHW;
#endif

#endif // end if MIO_BN_NGRPS

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
        _FLOAT NHW                  = (_FLOAT)MIO_BN_NHW;
        const _FLOAT adjust         = (MIO_BN_NHW == 1) ? variance : variance * (NHW / (NHW - 1));
        resultRunningVariance[xgid] = (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + xgid) +
                                      (_FLOAT)expAvgFactor * adjust;
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
        lcl_mean = *(meanvarbuff + meanstashindex); // load stashed mean
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
        mean = lcl_mean;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index   = n * MIO_BN_CHW + cidx + ygid;
            elemStd = (*(in + index) - mean);
            variance += elemStd * elemStd;
        }
    }

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&variance, lcl_data, ylid, 1);

#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, ylid, 1);

#endif

    if(ylid == 0)
    {
        unsigned int varindex = cidx + ygrp_sz * ygrp_id + 2;
        meanvarbuff[varindex] = variance; // pre-stage for group reduction
    }

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
            mean += *(meanvarbuff + meanindex);
        }
    }

#if(MIO_BN_NGRPS > 64)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&mean, lcl_data, ylid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, ylid, INHW);
#endif

#elif(MIO_BN_NGRPS <= 64)

#ifdef __AMDGCN__
    commitID = 63;
    dppRegReduce64(&mean, INHW);

#else
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    regLDSreduce(&mean, lcl_data, ylid, INHW);
    commitID = 0;
#endif
#else
#ifdef __AMDGCN__
    dppRegReduce16(&mean, INHW);
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
    mean *= INHW;
#endif
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
            mean += *(in + index);
        }
    }

    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&mean, lcl_data, ylid, 1);

#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, ylid, 1);

#endif
    if(ylid == 0)
    {
        meanbuff[meanindex] = mean;
    }
} // end spatial mean kernel

#elif(MIO_BN_VARIANT == 4)

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
    _FLOAT pvscale     = 0.;
    _FLOAT pvbias      = 0.;

    _FLOAT batchvalues[MIO_BN_N][MIO_BN_SEGMENT];

    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int index   = 0;
    unsigned int ylid    = get_local_id(1);
    unsigned int xgrpid  = get_group_id(0);
    unsigned int lidhw   = 0;
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int cid     = xgid * MIO_BN_HW;
    unsigned int nid     = 0;

    if(ylid == 0)
    {
        lcl_scale = *(scale + xgrpid);
        lcl_bias  = *(bias + xgrpid);
    }

// if(lid < MIO_BN_SEGMENT){
//==== CALC MEAN =======================
#pragma unroll
    for(unsigned int hw = 0; hw < MIO_BN_SEGMENT; hw++)
    {
        lidhw = hw * MIO_BN_GRP1 + ylid;
        if(lidhw < MIO_BN_HW)
        {
#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            {
                nid                = n * MIO_BN_CHW;
                index              = nid + cid + lidhw;
                batchvalues[n][hw] = *(in + index); //(lidhw < MIO_BN_HW) ? in[index] : 0.;
                mean += batchvalues[n][hw];
            }
        }
    }
    //}
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&mean, lcl_data, ylid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, ylid, INHW);
#endif

//==== CALC VARIANCE =======================
#pragma unroll
    for(unsigned int hw = 0; hw < MIO_BN_SEGMENT; hw++)
    {
        lidhw = hw * MIO_BN_GRP1 + ylid;
        if(lidhw < MIO_BN_HW)
        {
#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            {
                elemStd  = (batchvalues[n][hw] - mean);
                variance = mad(elemStd, elemStd, variance);
            }
        }
    }
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&variance, lcl_data, ylid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, ylid, INHW);
#endif

    invVariance = rsqrt(variance + epsilon);

    //==== CALC NORM =======================
    pvscale = lcl_scale;
    pvbias  = lcl_bias;

#pragma unroll
    for(unsigned int hw = 0; hw < MIO_BN_SEGMENT; hw++)
    {
        lidhw = hw * MIO_BN_GRP1 + ylid;
        if(lidhw < MIO_BN_HW)
        {
#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            { // apply normalization
                nid   = n * MIO_BN_CHW;
                inhat = (batchvalues[n][hw] - mean) * invVariance;
                index = nid + cid + lidhw;
                // if(index < MIO_BN_NCHW)
                out[index] = mad(pvscale, inhat, pvbias);
            }
        }
    } // end for
      //}
#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(ylid == 0)
    {

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[xgrpid]        = mean;
        resultSaveInvVariance[xgrpid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean            = resultRunningMean[xgrpid];
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[grpid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));

        resultRunningVariance[xgrpid] = (1 - (_FLOAT)expAvgFactor) * resultRunningVariance[xgrpid] +
                                        (_FLOAT)expAvgFactor * adjust;
#endif
    }
#endif
} // end spatial norm

#elif(MIO_BN_VARIANT == 5)

#define MIO_BN_REM (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_GRP0) * MIO_BN_GRP0))
#define MIO_BN_LESS (MIO_BN_NHW - MIO_BN_REM)

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
    _FLOAT elemStd     = 0.;
    _FLOAT pvscale     = elemStd;
    _FLOAT pvbias      = 0.;

    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int index = 0;
    unsigned int lid   = get_local_id(0);
    unsigned int grpid = get_group_id(0);
    unsigned int chwid = grpid * MIO_BN_HW;
    unsigned int nidx  = 0;
    unsigned int hwidx = 0;

    _FLOAT diff = 0.;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }

//==== CALC MEAN =======================
#pragma unroll
    for(unsigned int k = lid; k < MIO_BN_LESS; k += MIO_BN_GRP0)
    {
        nidx  = iDiv(k, MIO_BN_HW);
        hwidx = iMod(k, nidx, MIO_BN_HW);
        index = nidx * MIO_BN_CHW + chwid + hwidx;
        mean += *(in + index);
    }

#if(MIO_BN_REM)
    nidx  = iDiv(MIO_BN_LESS + lid, MIO_BN_HW);
    hwidx = iMod(MIO_BN_LESS + lid, nidx, MIO_BN_HW);
    index = nidx * MIO_BN_CHW + grpid * MIO_BN_HW + hwidx;
    mean += (index < MIO_BN_NCHW) ? *(in + index) : 0.;
#endif
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&mean, lcl_data, lid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, INHW);

#endif

    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for(unsigned int k = lid; k < MIO_BN_LESS; k += MIO_BN_GRP0)
    {
        nidx     = iDiv(k, MIO_BN_HW);
        hwidx    = iMod(k, nidx, MIO_BN_HW);
        index    = nidx * MIO_BN_CHW + chwid + hwidx;
        diff     = *(in + index) - mean;
        variance = mad(diff, diff, variance);
    }

#if(MIO_BN_REM)
    nidx  = iDiv(MIO_BN_LESS + lid, MIO_BN_HW);
    hwidx = iMod(MIO_BN_LESS + lid, nidx, MIO_BN_HW);
    index = nidx * MIO_BN_CHW + chwid + hwidx;
    if(index < MIO_BN_NCHW)
    {
        diff     = *(in + index) - mean;
        variance = mad(diff, diff, variance);
    }
#endif

    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&variance, lcl_data, lid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, INHW);
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
    invVariance = rsqrt(variance + epsilon);

    //==== CALC NORM =======================
    pvscale = lcl_scale;
    pvbias  = lcl_bias;

#pragma unroll
    for(unsigned int k = lid; k < MIO_BN_LESS; k += MIO_BN_GRP0)
    {

        nidx       = iDiv(k, MIO_BN_HW);
        hwidx      = iMod(k, nidx, MIO_BN_HW);
        index      = nidx * MIO_BN_CHW + chwid + hwidx;
        out[index] = mad(pvscale, (*(in + index) - mean) * invVariance, pvbias);
    } // end for
#if(MIO_BN_REM)
    nidx  = iDiv(MIO_BN_LESS + lid, MIO_BN_HW);
    hwidx = iMod(MIO_BN_LESS + lid, nidx, MIO_BN_HW);
    index = nidx * MIO_BN_CHW + chwid + hwidx;
    if(index < MIO_BN_NCHW)
    {
        *(out + index) = mad(pvscale, (*(in + index) - mean) * invVariance, pvbias);
    }
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(lid == 0)
    {
// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[grpid]        = mean;
        resultSaveInvVariance[grpid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean           = *(resultRunningMean + grpid);
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[grpid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        resultRunningVariance[grpid] =
            (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + grpid) +
            (_FLOAT)expAvgFactor * adjust;
#endif
    }
#endif
}

#elif(MIO_BN_VARIANT == 6)

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
    _FLOAT elemStd     = 0.;
    _FLOAT pvscale     = elemStd;
    _FLOAT pvbias      = 0.;

    __local _FLOAT lcl_bias;
    __local _FLOAT lcl_scale;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int index = 0;
    unsigned int lid   = get_local_id(0);
    unsigned int grpid = get_group_id(0);
    unsigned int chwid = grpid * MIO_BN_HW;

    _FLOAT diff = 0.;

    if(lid == 0)
    {
        lcl_scale = *(scale + grpid);
        lcl_bias  = *(bias + grpid);
    }

//==== CALC MEAN =======================
#pragma unroll
    for(unsigned int n = 0; n < MIO_BN_N; n++)
    {
        for(unsigned int hw = lid; hw < MIO_BN_HW; hw += MIO_BN_GRP0)
        {

            index = n * MIO_BN_CHW + chwid + hw;
            mean += (index < MIO_BN_NCHW) ? *(in + index) : 0.;
        }
    }
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&mean, lcl_data, lid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, INHW);

#endif

    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for(unsigned int n = 0; n < MIO_BN_N; n++)
    {
        for(unsigned int hw = lid; hw < MIO_BN_HW; hw += MIO_BN_GRP0)
        {
            index    = n * MIO_BN_CHW + chwid + hw;
            diff     = ((index < MIO_BN_NCHW) ? *(in + index) : 0.) - mean;
            variance = mad(diff, diff, variance);
        }
    }

    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&variance, lcl_data, lid, INHW);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, INHW);
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
    invVariance = rsqrt(variance + epsilon);

    //==== CALC NORM =======================
    pvscale = lcl_scale;
    pvbias  = lcl_bias;

#pragma unroll
    for(unsigned int n = 0; n < MIO_BN_N; n++)
    {
        for(unsigned int hw = lid; hw < MIO_BN_HW; hw += MIO_BN_GRP0)
        {
            index      = n * MIO_BN_CHW + chwid + hw;
            _FLOAT tmp = (((index < MIO_BN_NCHW) ? *(in + index) : 0.) - mean) * invVariance;
            out[index] = mad(pvscale, tmp, pvbias);
        }
    } // end for

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(lid == 0)
    {
// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        resultSaveMean[grpid]        = mean;
        resultSaveInvVariance[grpid] = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT pvt_runMean           = *(resultRunningMean + grpid);
        _FLOAT pvt_newRunMean =
            mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[grpid] =
            mad(mean, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT adjust = (MIO_BN_NHW == 1)
                                  ? variance
                                  : variance * ((_FLOAT)MIO_BN_NHW / (_FLOAT)(MIO_BN_NHW - 1.0));
        resultRunningVariance[grpid] =
            (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + grpid) +
            (_FLOAT)expAvgFactor * adjust;
#endif
    }
#endif
}

#endif

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
