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

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#define _FLOAT_ACCUM float
#if MIOPEN_USE_FP16 == 1
#define MIO_BN_NODPP 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_PREC half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define _FLOAT_PREC float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif
#if MIOPEN_USE_FPMIX == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_PREC float
/*
#ifndef HALF_MAX
#define MAX_VAL 65504
#else
#define MAX_VAL HALF_MAX
#endif
*/
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)
#define _AS_FLOAT PPCAT(as_, _FLOAT)

#ifndef MIO_BN_LDSGCN_SIZE
#define MIO_BN_LDSGCN_SIZE 16
#endif

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 256
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

#ifndef MIO_BN_NCHW
#define MIO_BN_NCHW 1
#endif

#ifndef MIO_BN_VARIANT
#define MIO_BN_VARIANT 255
#endif

#ifndef MIO_BN_MAXN
#define MIO_BN_MAXN 65
#endif

#ifndef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#elif(MIO_BN_NODPP == 1 && MIO_BN_VARIANT != 0)
#undef __AMDGCN__
#endif

/*#ifdef __AMDGCN__
#undef __AMDGCN__
#endif*/

#define UNUSED __attribute__((__unused__))

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

#ifndef __AMDGCN__
#if MIOPEN_USE_FP16 == 1
static inline void ReduceKernel(__local float* lcl_mem,
                                unsigned int sum_stride,
                                unsigned int unit_id,
                                unsigned int unit_len)
{
    float sum               = (float)0.;
    unsigned int lcl_offset = unit_id * unit_len;

    for(unsigned int i = 0; i < unit_len; i += sum_stride)
    {
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}

static inline void
regLDSreduce(float* value, __local float* data, unsigned int localID, float scale)
#else
static inline void ReduceKernel(__local _FLOAT* lcl_mem,
                                unsigned int sum_stride,
                                unsigned int unit_id,
                                unsigned int unit_len)
{
    _FLOAT sum              = (_FLOAT)0.;
    unsigned int lcl_offset = unit_id * unit_len;

    __attribute__((opencl_unroll_hint(2))) for(unsigned int i = 0; i < unit_len; i += sum_stride)
    {
        sum += lcl_mem[lcl_offset + i];
    }
    lcl_mem[lcl_offset] = sum;
}

static inline void
regLDSreduce(_FLOAT* value, __local _FLOAT* data, unsigned int localID, _FLOAT scale)
#endif
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
#endif

#ifdef __AMDGCN__

static inline void dpp_reduction(_FLOAT_ACCUM* temp_sum)
{
    __asm__ volatile("s_nop 4\n"
                     "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                     "s_nop 1\n"
                     "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                     "s_nop 1\n"
                     : "=v"(*temp_sum)
                     : "0"(*temp_sum));
}

static inline void dpp_interleaved_reduction(_FLOAT_ACCUM* temp_sum1, _FLOAT_ACCUM* temp_sum2)
{
    __asm__ volatile("s_nop 4\n"
                     "v_add_f32 %0 %0 %0 row_shr:1 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:1 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:2 bound_ctrl:0\n"
                     "v_add_f32 %1 %1 %1 row_shr:2 bound_ctrl:0\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:4 bank_mask:0xe\n"
                     "v_add_f32 %1 %1 %1 row_shr:4 bank_mask:0xe\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_shr:8 bank_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_shr:8 bank_mask:0xc\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:15 row_mask:0xa\n"
                     "v_add_f32 %1 %1 %1 row_bcast:15 row_mask:0xa\n"
                     "s_nop 0\n"
                     "v_add_f32 %0 %0 %0 row_bcast:31 row_mask:0xc\n"
                     "v_add_f32 %1 %1 %1 row_bcast:31 row_mask:0xc\n"
                     "s_nop 0"
                     : "=v"(*temp_sum1), "=v"(*temp_sum2)
                     : "0"(*temp_sum1), "1"(*temp_sum2));
}

#endif

#if(MIO_BN_VARIANT == 0)

#define MIO_BN_SEGTMP (MIO_BN_HW * (MIO_BN_GRP0 / MIO_BN_HW))
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
    _FLOAT_ACCUM batchvalues[MIO_BN_NLOOP];

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
#if MIOPEN_USE_FP16 == 0
        __attribute__((opencl_unroll_hint(2)))
#endif
        for(unsigned int n = 0; n < MIO_BN_NLOOPM; ++n)
        {
            nid            = n * MIO_BN_SEGIHW + lidihw;
            index          = nid * MIO_BN_CHW + chwid;
            batchvalues[n] = (_FLOAT_ACCUM)(*(in + index));
            mean += batchvalues[n];
            variance = mad(batchvalues[n], batchvalues[n], variance);
        }
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        batchvalues[MIO_BN_NLOOPM] =
            (index < MIO_BN_NCHW) ? (_FLOAT_ACCUM)(*(in + index)) : (_FLOAT_ACCUM)0.;
        mean += batchvalues[MIO_BN_NLOOPM];
        variance = mad(batchvalues[MIO_BN_NLOOPM], batchvalues[MIO_BN_NLOOPM], variance);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#ifndef __AMDGCN__
    __local _FLOAT_ACCUM lcl_data[MIO_BN_LDS_SIZE];

    // Reduce mean
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, (_FLOAT_ACCUM)INHW);
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
    regLDSreduce(&variance, lcl_data, lid, (_FLOAT_ACCUM)INHW);
    barrier(CLK_LOCAL_MEM_FENCE);
#else

    unsigned int ldsidx = lid >> 6;
    __local _FLOAT_ACCUM lcl_mean[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT_ACCUM lcl_variance[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&mean, &variance);

    if((lid % 64) == 63)
    {
        lcl_mean[ldsidx]     = mean;
        lcl_variance[ldsidx] = variance;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    mean = variance = (_FLOAT_ACCUM)0.;
    __attribute__((opencl_unroll_hint(2))) for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        mean += lcl_mean[i];
        variance += lcl_variance[i];
    }
    mean *= (_FLOAT_ACCUM)INHW;
    variance *= (_FLOAT_ACCUM)INHW;

#endif

    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + (_FLOAT_ACCUM)epsilon);
    pvscale     = (_FLOAT_ACCUM)lcl_scale;
    pvbias      = (_FLOAT_ACCUM)lcl_bias;

    if(lid < MIO_BN_SEGMENT)
    {
        //==== CALC NORM =======================
        _FLOAT_ACCUM inhat = (_FLOAT_ACCUM)0.;

#if MIOPEN_USE_FP16 == 0
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

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(lid == 0)
    {

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        *(resultSaveMean + grpid)        = (_FLOAT_PREC)mean;
        *(resultSaveInvVariance + grpid) = (_FLOAT_PREC)invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT_ACCUM pvt_runMean    = (_FLOAT_ACCUM)(*(resultRunningMean + grpid));
        _FLOAT_ACCUM pvt_newRunMean = mad(
            (_FLOAT_ACCUM)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[grpid] = (_FLOAT_PREC)mad(
            mean, (_FLOAT_ACCUM)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
        const _FLOAT_ACCUM adjust =
            (MIO_BN_NHW == 1) ? variance
                              : variance * ((_FLOAT_ACCUM)MIO_BN_NHW /
                                            ((_FLOAT_ACCUM)MIO_BN_NHW - (_FLOAT_ACCUM)1.0));

        resultRunningVariance[grpid] = (_FLOAT_PREC)(
            (1 - (_FLOAT_ACCUM)expAvgFactor) * (_FLOAT_ACCUM)(*(resultRunningVariance + grpid)) +
            (_FLOAT_ACCUM)expAvgFactor * adjust);
#endif
    }
#endif
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
    uint chwid = grpid * MIO_BN_HW;
    uint nidx  = 0;
    uint hwidx = 0;

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
        _FLOAT_PREC xin = (index < MIO_BN_NCHW) ? (_FLOAT_PREC)(*(in + index)) : (_FLOAT_PREC)0.;
        mean += xin;
        variance = mad(xin, xin, variance);
    }
#endif
#endif
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
// REDUCE MEAN AND VARIANCE -----------------------
#ifndef __AMDGCN__
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
    local _FLOAT_PREC lcl_data[MIO_BN_LDS_SIZE];
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
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    unsigned int ldsidx = lid >> 6;
    __local _FLOAT_PREC lcl_mean[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT_PREC lcl_variance[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&mean, &variance);

    if((lid % 64) == 63)
    {
        lcl_mean[ldsidx]     = mean;
        lcl_variance[ldsidx] = variance;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = variance = (_FLOAT_PREC)0.;
    __attribute__((opencl_unroll_hint(2))) for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        mean += lcl_mean[i];
        variance += lcl_variance[i];
    }
    mean *= (_FLOAT_PREC)INHW;
    variance *= (_FLOAT_PREC)INHW;
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
        nidx  = k / MIO_BN_HW;
        hwidx = k - (nidx * MIO_BN_HW);
        index = nidx * MIO_BN_CHW + chwid + hwidx;
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

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(lid == 0)
    {

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        *(resultSaveMean + grpid)        = mean;
        *(resultSaveInvVariance + grpid) = invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT_PREC pvt_runMean          = (_FLOAT_PREC)(*(resultRunningMean + grpid));
        _FLOAT_PREC pvt_newRunMean       = mad(
            (_FLOAT_PREC)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[grpid] =
            mad(mean, (_FLOAT_PREC)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
#if MIOPEN_USE_FP16 == 1
        const float temp_adjust =
            (MIO_BN_NHW == 1) ? ((float)variance)
                              : ((float)variance) * ((float)MIO_BN_NHW / ((float)MIO_BN_NHW - 1.0));
        const _FLOAT_PREC adjust = (_FLOAT_PREC)temp_adjust;
#else
        const _FLOAT_PREC adjust = (MIO_BN_NHW == 1)
                                       ? variance
                                       : variance * ((_FLOAT_PREC)MIO_BN_NHW /
                                                     ((_FLOAT_PREC)MIO_BN_NHW - (_FLOAT_PREC)1.0));
#endif
        resultRunningVariance[grpid] =
            (1 - (_FLOAT_PREC)expAvgFactor) * *(resultRunningVariance + grpid) +
            (_FLOAT_PREC)expAvgFactor * adjust;
#endif
    }
#endif
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
        __attribute__((opencl_unroll_hint(2))) for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index        = n * MIO_BN_CHW + cidx + ygid;
            _FLOAT inhat = (*(in + index) - mean) * invVariance;
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
#ifndef __AMDGCN__
#if MIOPEN_USE_FP16 == 1
    __local float lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[lid] = (float)mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    float temp_mean = (float)mean;
#if(MIO_BN_NGRPS > 64)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&temp_mean, lcl_data, lid, (float)INHW);
#elif(MIO_BN_NGRPS <= 64)
    regLDSreduce(&temp_mean, lcl_data, lid, (float)INHW);
    commitID = 0;
#else
    temp_mean = (float)0.;
    __attribute__((opencl_unroll_hint(2))) for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        temp_mean += lcl_data[i];
    }

#endif
    mean = (_FLOAT)temp_mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = (float)variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    float temp_variance = (float)variance;
#if(MIO_BN_NGRPS > 256)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&temp_variance, lcl_data, lid, (float)INHW);
#elif(MIO_BN_NGRPS > 64)
    regLDSreduce(&temp_variance, lcl_data, lid, (float)INHW);
#elif(MIO_BN_NGRPS > 16)
    regLDSreduce(&temp_variance, lcl_data, lid, (float)INHW);
#else //(MIO_BN_NGRPS <= 16)
    temp_variance = (float)0.;
    __attribute__((opencl_unroll_hint(2))) for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        temp_variance += lcl_data[i];
    }
#endif
    variance = (_FLOAT)temp_variance;
#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[lid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_NGRPS > 64)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, (_FLOAT)INHW);
#elif(MIO_BN_NGRPS <= 64)
    regLDSreduce(&mean, lcl_data, lid, (_FLOAT)INHW);
    commitID = 0;
#else
    mean = (_FLOAT)0.;
    __attribute__((opencl_unroll_hint(2))) for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        mean += lcl_data[i];
    }

#endif

    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_NGRPS > 256)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, (_FLOAT)INHW);
#elif(MIO_BN_NGRPS > 64)
    regLDSreduce(&variance, lcl_data, lid, (_FLOAT)INHW);
#elif(MIO_BN_NGRPS > 16)
    regLDSreduce(&variance, lcl_data, lid, (_FLOAT)INHW);
#else //(MIO_BN_NGRPS <= 16)
    variance = (_FLOAT)0.;
    __attribute__((opencl_unroll_hint(2))) for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        variance += lcl_data[i];
    }
#endif

#endif

#else
    commitID            = 64;
    unsigned int ldsidx = lid >> 6;
    __local _FLOAT lcl_mean[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT lcl_variance[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&mean, &variance);

    if((lid % 64) == 63)
    {
        lcl_mean[ldsidx]     = mean;
        lcl_variance[ldsidx] = variance;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    mean = variance = 0.;

    __attribute__((opencl_unroll_hint(2))) for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        mean += lcl_mean[i];
        variance += lcl_variance[i];
    }
    mean *= (_FLOAT)INHW;
    variance *= (_FLOAT)INHW;
#endif
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + epsilon);
    if(lid == commitID)
    {
        meanvarbuff[meanstashindex] = mean;        // stash mean
        meanvarbuff[varstashindex]  = invVariance; // stash mean
    }

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    unsigned int ygid = get_global_id(1);
    if(ygid == commitID)
    {
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

#if MIOPEN_USE_FP16 == 1
        const float temp_adjust =
            (MIO_BN_NHW == 1) ? ((float)variance)
                              : ((float)variance) * ((float)MIO_BN_NHW / (float)(MIO_BN_NHW - 1.0));
        const _FLOAT adjust         = (_FLOAT)temp_adjust;
#else
        _FLOAT NHW          = (_FLOAT)MIO_BN_NHW;
        const _FLOAT adjust = (MIO_BN_NHW == 1) ? variance : variance * (NHW / (NHW - 1));
#endif
        resultRunningVariance[xgid] = (1 - (_FLOAT)expAvgFactor) * *(resultRunningVariance + xgid) +
                                      (_FLOAT)expAvgFactor * adjust;
#endif
    }
#endif
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
        __attribute__((opencl_unroll_hint(2))) for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            value = *(in + index);
            mean += value;
            variance = mad(value, value, variance);
        }
    }

#ifdef __AMDGCN__
    unsigned int ldsidx = ylid >> 6;
    __local _FLOAT lcl_mean[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT lcl_variance[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&mean, &variance);

    if((ylid % 64) == 63)
    {
        lcl_mean[ldsidx]     = mean;
        lcl_variance[ldsidx] = variance;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    mean = variance = 0.;

    __attribute__((opencl_unroll_hint(2))) for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        mean += lcl_mean[i];
        variance += lcl_variance[i];
    }

#else
#if MIOPEN_USE_FP16 == 1
    local float lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = (float)mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_mean = (float)mean;
    regLDSreduce(&temp_mean, lcl_data, ylid, 1);
    mean = (_FLOAT)temp_mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[ylid] = (float)variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_variance = (float)variance;
    regLDSreduce(&temp_variance, lcl_data, ylid, 1);
    variance = (_FLOAT)temp_variance;
#else
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, ylid, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, ylid, 1);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
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
    _FLOAT_PREC pvscale, pvbias;

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
#ifndef __AMDGCN__
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
    barrier(CLK_LOCAL_MEM_FENCE);
#else

    unsigned int ldsidx = lid >> 6;
    __local _FLOAT_PREC lcl_mean[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT_PREC lcl_variance[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&mean, &variance);

    if((lid % 64) == 63)
    {
        lcl_mean[ldsidx]     = mean;
        lcl_variance[ldsidx] = variance;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = variance = (_FLOAT_PREC)0.;
    __attribute__((opencl_unroll_hint(2))) for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        mean += lcl_mean[i];
        variance += lcl_variance[i];
    }
    mean *= (_FLOAT_PREC)INHW;
    variance *= (_FLOAT_PREC)INHW;

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
            index      = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            inhat      = (minibatch[n] - mean) * invVariance; // (in[index] - mean) * invVariance;
#else
            inhat = ((_FLOAT_PREC)(*(in + index)) - mean) * invVariance;
// printf("lid: %d, index: %d, n: %d, mean: %f, invVar: %f\n", lid, index, n, mean, invVariance);
#endif
            out[index] = (_FLOAT)mad(pvscale, inhat, pvbias);
        } // end for
    }     // end if

#if(MIO_SAVE_MEAN_VARIANCE == 1 || MIO_RUNNING_RESULT == 1)
    if(lid == 0)
    {

// Save mean and calculate and save running mean
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        *(resultSaveMean + grpid)        = (_FLOAT_PREC)mean;
        *(resultSaveInvVariance + grpid) = (_FLOAT_PREC)invVariance;
#endif

#if(MIO_RUNNING_RESULT == 1)
        _FLOAT_PREC pvt_runMean          = (_FLOAT_PREC)(*(resultRunningMean + grpid));
        _FLOAT_PREC pvt_newRunMean       = mad(
            (_FLOAT_PREC)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
        resultRunningMean[grpid] =
            mad(mean, (_FLOAT_PREC)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
#if MIOPEN_USE_FP16 == 1
        const float temp_adjust =
            (MIO_BN_NHW == 1) ? ((float)variance)
                              : ((float)variance) * ((float)MIO_BN_NHW / ((float)MIO_BN_NHW - 1.0));
        const _FLOAT_PREC adjust = (_FLOAT_PREC)temp_adjust;
#else
        const _FLOAT_PREC adjust = (MIO_BN_NHW == 1)
                                       ? variance
                                       : variance * ((_FLOAT_PREC)MIO_BN_NHW /
                                                     ((_FLOAT_PREC)MIO_BN_NHW - (_FLOAT_PREC)1.0));
#endif
        resultRunningVariance[grpid] =
            (1 - (_FLOAT_PREC)expAvgFactor) * *(resultRunningVariance + grpid) +
            (_FLOAT_PREC)expAvgFactor * adjust;
#endif
    }
#endif
} // end spatial norm

#endif

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
