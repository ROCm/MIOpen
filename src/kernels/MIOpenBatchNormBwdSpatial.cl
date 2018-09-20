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

#define _FLOAT_PREC4 PPCAT(_FLOAT_PREC, FOUR)

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 1
#endif

#ifndef MIO_BN_LDSGCN_SIZE
#define MIO_BN_LDSGCN_SIZE 16
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

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
#endif

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

#ifndef MIO_BN_NCHW
#define MIO_BN_NCHW 1
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

#ifndef MIO_BN_USESAVED
#define MIO_BN_USESAVED 1
#endif

#ifndef MIO_BN_MAXN
#define MIO_BN_MAXN 65
#endif

#ifndef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#elif(MIO_BN_NODPP == 1)
#undef __AMDGCN__
#endif

/*#ifdef __AMDGCN__
#undef __AMDGCN__
#endif*/

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

#define UNUSED __attribute__((__unused__))

#ifndef __AMDGCN__
#if MIOPEN_USE_FP16 == 1
static inline void ReduceKernel(__local float* lcl_mem,
                                unsigned int sum_stride,
                                unsigned int unit_id,
                                unsigned int unit_len)
{
    float sum               = (float)0;
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
    _FLOAT sum              = (_FLOAT)0;
    unsigned int lcl_offset = unit_id * unit_len;

    for(unsigned int i = 0; i < unit_len; i += sum_stride)
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

static inline void dpp_reduction(_FLOAT_PREC* temp_sum)
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

static inline void dpp_interleaved_reduction(_FLOAT_PREC* temp_sum1, _FLOAT_PREC* temp_sum2)
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
                     "s_nop 1"
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
MIOpenBatchNormBwdSpatial(const __global _FLOAT* __restrict x_in,
                          const __global _FLOAT* __restrict dy_in,
                          __global _FLOAT* __restrict dx_out,
                          const __global _FLOAT_PREC* bnScale,
                          __global _FLOAT_PREC* __restrict dscale,
                          __global _FLOAT_PREC* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                          double epsilon,
#elif(MIO_BN_USESAVED == 1)
                          const __global _FLOAT_PREC* savedMean,
                          const __global _FLOAT_PREC* savedInvVariance,
#endif
                          _FLOAT_PREC INHW)
{

    // SPATIAL
    _FLOAT_PREC mean = (_FLOAT_PREC)0.;
#if(MIO_BN_USESAVED == 0)
    _FLOAT_PREC variance = (_FLOAT_PREC)0.;
#endif
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pscale      = (_FLOAT_PREC)0.;
    _FLOAT_PREC ds          = (_FLOAT_PREC)0.;
    _FLOAT_PREC db          = (_FLOAT_PREC)0.;

    _FLOAT_PREC batchvalues[MIO_BN_NLOOP];
    _FLOAT_PREC dyvalues[MIO_BN_NLOOP];

    __local _FLOAT_PREC lbns;

#if(MIO_BN_USESAVED == 1)
    __local _FLOAT_PREC lmean, lvar;
#endif
    unsigned int index  = 0;
    unsigned int lid    = get_local_id(0);
    unsigned int grpid  = get_group_id(0);
    unsigned int chwid  = grpid * MIO_BN_HW + (lid % MIO_BN_HW);
    unsigned int lidihw = lid / MIO_BN_HW;
    unsigned int nid    = 0;
    _FLOAT_PREC tmp1, tmp2, tmp3;

    _FLOAT_PREC NHW = (_FLOAT_PREC)MIO_BN_NHW;

    if(lid == 0)
    {
        lbns = *(bnScale + grpid);

#if(MIO_BN_USESAVED == 1)
        lmean = *(savedMean + grpid);
        lvar  = *(savedInvVariance + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean        = lmean;
    invVariance = lvar;
#else // recalc mean and variance below
    } // end if(!lid)

    // == RECALC MEAN AND VARIANCE ===========
    if(lid < MIO_BN_SEGMENT)
    {

        for(unsigned int n = 0; n < MIO_BN_NLOOPM; ++n)
        {
            nid            = n * MIO_BN_SEGIHW + lidihw;
            index          = nid * MIO_BN_CHW + chwid;
            batchvalues[n] = (_FLOAT_PREC)(*(x_in + index));
            mean += batchvalues[n];
            variance = mad(batchvalues[n], batchvalues[n], variance);
        }
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        batchvalues[MIO_BN_NLOOPM] =
            (index < MIO_BN_NCHW) ? (_FLOAT_PREC)(*(x_in + index)) : (_FLOAT_PREC)0.;
        mean += batchvalues[MIO_BN_NLOOPM];
        variance = mad(batchvalues[MIO_BN_NLOOPM], batchvalues[MIO_BN_NLOOPM], variance);
    }

#ifndef __AMDGCN__
#if MIOPEN_USE_FP16 == 1
    __local float lcl_data[MIO_BN_LDS_SIZE];

    // Reduce mean
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

    // Reduce variance
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

    __local _FLOAT_PREC lcl_mean[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT_PREC lcl_variance[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&mean, &variance);
    if((lid % 64) == 63)
    {
        unsigned int ldsidx  = lid >> 6;
        lcl_mean[ldsidx]     = mean;
        lcl_variance[ldsidx] = variance;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = variance = (_FLOAT_PREC)0.;

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        mean += lcl_mean[i];
        variance += lcl_variance[i];
    }
    mean *= (_FLOAT_PREC)INHW;
    variance *= (_FLOAT_PREC)INHW;

#endif

    variance               = mad(-mean, mean, variance);
    invVariance            = rsqrt(variance + epsilon);
#endif // end -- Recalc mean and variance
    //-------------------------------------------

    //==== CALC DB and DS =========================================
    if(lid < MIO_BN_SEGMENT)
    {

        for(unsigned int n = 0; n < MIO_BN_NLOOPM; ++n)
        {
            nid         = n * MIO_BN_SEGIHW + lidihw;
            index       = nid * MIO_BN_CHW + chwid;
            dyvalues[n] = (_FLOAT_PREC)(*(dy_in + index));
            db += dyvalues[n];

#if(MIO_BN_USESAVED == 1)
            batchvalues[n] = ((_FLOAT_PREC)(*(x_in + index)) - mean) * invVariance;
#else
            batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif
            // batchvalues is now xhat
            ds = mad(batchvalues[n], dyvalues[n], ds);
        }
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        dyvalues[MIO_BN_NLOOPM] =
            ((index < MIO_BN_NCHW) ? (_FLOAT_PREC)(*(dy_in + index)) : (_FLOAT_PREC)0.);
        db += dyvalues[MIO_BN_NLOOPM];

#if(MIO_BN_USESAVED == 1)
        batchvalues[MIO_BN_NLOOPM] = (index < MIO_BN_NCHW)
                                         ? (((_FLOAT_PREC)(*(x_in + index)) - mean) * invVariance)
                                         : (_FLOAT_PREC)0.;

#else
        batchvalues[MIO_BN_NLOOPM] = (batchvalues[MIO_BN_NLOOPM] - mean) * invVariance;
#endif
        // batchvalues is now xhat
        ds = mad(batchvalues[MIO_BN_NLOOPM], dyvalues[MIO_BN_NLOOPM], ds);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#ifndef __AMDGCN__
#if MIOPEN_USE_FP16 == 1
#if(MIO_BN_USESAVED == 1)
    __local float lcl_data[MIO_BN_LDS_SIZE];
#endif
    lcl_data[lid] = (float)ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_ds = (float)ds;
    regLDSreduce(&temp_ds, lcl_data, lid, (float)1.0);
    ds = (_FLOAT_PREC)temp_ds;
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[lid] = (float)db;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_db = (float)db;
    regLDSreduce(&temp_db, lcl_data, lid, (float)1.0);
    db = (_FLOAT_PREC)temp_db;
#else
#if(MIO_BN_USESAVED == 1)
    __local _FLOAT_PREC lcl_data[MIO_BN_LDS_SIZE];
#endif
    lcl_data[lid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&ds, lcl_data, lid, (_FLOAT_PREC)1.0);
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[lid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&db, lcl_data, lid, (_FLOAT_PREC)1.0);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
#else

    __local _FLOAT_PREC lcl_ds[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT_PREC lcl_db[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&ds, &db);
    if((lid % 64) == 63)
    {
        unsigned int ldsidx = lid >> 6;
        lcl_ds[ldsidx]      = ds;
        lcl_db[ldsidx]      = db;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = db = (_FLOAT_PREC)0.;

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        ds += lcl_ds[i];
        db += lcl_db[i];
    }
#endif

    if(lid < MIO_BN_SEGMENT)
    {
        //==== CALC NORM =======================
        pscale = lbns;

        for(unsigned int n = 0; n < MIO_BN_NLOOPM; n++)
        { // apply normalization
            nid           = n * MIO_BN_SEGIHW + lidihw;
            index         = nid * MIO_BN_CHW + chwid;
            tmp1          = mad(NHW, dyvalues[n], -db);
            tmp2          = -batchvalues[n] * ds;
            tmp3          = (pscale * invVariance) * INHW;
            dx_out[index] = (_FLOAT)(tmp3 * (tmp2 + tmp1));
        } // end for
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        if(index < MIO_BN_NCHW)
        {
            tmp1          = mad(NHW, dyvalues[MIO_BN_NLOOPM], -db);
            tmp2          = -batchvalues[MIO_BN_NLOOPM] * ds;
            tmp3          = (pscale * invVariance) * INHW;
            dx_out[index] = (_FLOAT)(tmp3 * (tmp2 + tmp1));
        }
    }
    if(lid == 0)
    {
        dbias[grpid]  = db;
        dscale[grpid] = ds;
    }
} // end spatial

#elif(MIO_BN_VARIANT == 1)

#define MIO_MAX_READ 2
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
MIOpenBatchNormBwdSpatial(const __global _FLOAT* __restrict x_in,
                          const __global _FLOAT* __restrict dy_in,
                          __global _FLOAT* __restrict dx_out,
                          const __global _FLOAT_PREC* bnScale,
                          __global _FLOAT_PREC* __restrict dscale,
                          __global _FLOAT_PREC* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                          double epsilon,
#elif(MIO_BN_USESAVED == 1)
                          const __global _FLOAT_PREC* savedMean,
                          const __global _FLOAT_PREC* savedInvVariance,
#endif
                          _FLOAT_PREC INHW)
{

    // SPATIAL
    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pscale      = (_FLOAT_PREC)0.;
    _FLOAT_PREC db          = (_FLOAT_PREC)0.;
    _FLOAT_PREC ds          = (_FLOAT_PREC)0.;
    _FLOAT_PREC xhat        = (_FLOAT_PREC)0.;
    _FLOAT_PREC dyvalue     = (_FLOAT_PREC)0.;

#if(MIO_BN_USESAVED == 1)
    __local _FLOAT_PREC lmean, lvar;
#endif

    __local _FLOAT_PREC lcl_scale;
    _FLOAT_PREC NHW = (_FLOAT_PREC)MIO_BN_NHW;

    unsigned int index = 0;
    unsigned int lid   = get_local_id(0);
    unsigned int grpid = get_group_id(0);
    unsigned int chwid = grpid * MIO_BN_HW;
    unsigned int nidx  = 0;
    unsigned int hwidx = 0;

    if(lid == 0)
    {
        lcl_scale = *(bnScale + grpid);
#if(MIO_BN_USESAVED == 1)
        lmean     = *(savedMean + grpid);
        lvar      = *(savedInvVariance + grpid);
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_USESAVED == 0)
    //==== CALC MEAN and VARIANCE ONCE AGAIN =======================
    _FLOAT_PREC variance = (_FLOAT_PREC)0.;
#if(MIO_BN_HW >= 4096)
    _FLOAT4 read4;
    __attribute__((opencl_unroll_hint(2))) for(unsigned int k = lid << 2; k < MIO_BN_LESS4;
                                               k += GRPRD)
    {
        nidx  = k / MIO_BN_HW;
        hwidx = k - (nidx * MIO_BN_HW);
        index = nidx * MIO_BN_CHW + chwid + hwidx;
        read4 = *((const global _FLOAT4*)(x_in + index));
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
    if(lid < MIO_BN_REM4)
    {
        unsigned int remkey = lid + MIO_BN_LESS4;
        nidx                = remkey / MIO_BN_HW;
        hwidx               = remkey - (nidx * MIO_BN_HW);
        index               = nidx * MIO_BN_CHW + chwid + hwidx;
        if(index < MIO_BN_NCHW)
        {
            read4 = *((const global _FLOAT4*)(x_in + index));
            mean += (_FLOAT_PREC)read4.x;
            mean += (_FLOAT_PREC)read4.y;
            mean += (_FLOAT_PREC)read4.z;
            mean += (_FLOAT_PREC)read4.w;
            variance = mad((_FLOAT_PREC)read4.x, (_FLOAT_PREC)read4.x, variance);
            variance = mad((_FLOAT_PREC)read4.y, (_FLOAT_PREC)read4.y, variance);
            variance = mad((_FLOAT_PREC)read4.z, (_FLOAT_PREC)read4.z, variance);
            variance = mad((_FLOAT_PREC)read4.w, (_FLOAT_PREC)read4.w, variance);
        }
    }
#endif
#else
    for(unsigned int k = lid; k < MIO_BN_LESS; k += MIO_BN_GRP0)
    {
        nidx           = k / MIO_BN_HW;
        hwidx          = k - (nidx * MIO_BN_HW);
        index          = nidx * MIO_BN_CHW + chwid + hwidx;
        _FLOAT_PREC in = (_FLOAT_PREC)(*(x_in + index));
        mean += in;
        variance = mad(in, in, variance);
    }
#if(MIO_BN_REM)
    if(lid < MIO_BN_REM)
    {
        unsigned int remkey = lid + MIO_BN_LESS;
        nidx                = remkey / MIO_BN_HW;
        hwidx               = remkey - (nidx * MIO_BN_HW);
        index               = nidx * MIO_BN_CHW + chwid + hwidx;
        _FLOAT_PREC in = (index < MIO_BN_NCHW) ? (_FLOAT_PREC)(*(x_in + index)) : (_FLOAT_PREC)0.;
        mean += in;
        variance = mad(in, in, variance);
    }
#endif // end REM
#endif // end if 4096

    barrier(CLK_LOCAL_MEM_FENCE);
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

    __local _FLOAT_PREC lcl_mean[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT_PREC lcl_variance[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&mean, &variance);
    if((lid % 64) == 63)
    {
        unsigned int ldsidx  = lid >> 6;
        lcl_mean[ldsidx]     = mean;
        lcl_variance[ldsidx] = variance;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = variance = (_FLOAT_PREC)0.;

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
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

#else // MIO_BN_USESAVED == 1

    mean        = lmean;
    invVariance = lvar;

#endif

    _FLOAT4 dyRead4;
    _FLOAT4 xread4;
    _FLOAT_PREC4 xhat4;
    __attribute__((opencl_unroll_hint(2))) for(unsigned int k = lid << 2; k < MIO_BN_LESS4;
                                               k += GRPRD)
    {
        nidx    = k / MIO_BN_HW;
        hwidx   = k - (nidx * MIO_BN_HW);
        index   = nidx * MIO_BN_CHW + chwid + hwidx;
        xread4  = *((const global _FLOAT4*)(x_in + index));
        dyRead4 = *((const global _FLOAT4*)(dy_in + index));
        xhat4.x = ((_FLOAT_PREC)xread4.x - mean) * invVariance;
        xhat4.y = ((_FLOAT_PREC)xread4.y - mean) * invVariance;
        xhat4.z = ((_FLOAT_PREC)xread4.z - mean) * invVariance;
        xhat4.w = ((_FLOAT_PREC)xread4.w - mean) * invVariance;
        db += (_FLOAT_PREC)dyRead4.x;
        db += (_FLOAT_PREC)dyRead4.y;
        db += (_FLOAT_PREC)dyRead4.z;
        db += (_FLOAT_PREC)dyRead4.w;
        ds = mad(xhat4.x, (_FLOAT_PREC)dyRead4.x, ds);
        ds = mad(xhat4.y, (_FLOAT_PREC)dyRead4.y, ds);
        ds = mad(xhat4.z, (_FLOAT_PREC)dyRead4.z, ds);
        ds = mad(xhat4.w, (_FLOAT_PREC)dyRead4.w, ds);
    }

#if(MIO_BN_REM4)
    unsigned int remkey = (lid << 2) + MIO_BN_LESS4;
    nidx                = remkey / MIO_BN_HW;
    hwidx               = remkey - (nidx * MIO_BN_HW);
    index               = nidx * MIO_BN_CHW + chwid + hwidx;
    if(index < MIO_BN_NCHW)
    {
        xread4  = *((const global _FLOAT4*)(x_in + index));
        dyRead4 = *((const global _FLOAT4*)(dy_in + index));
        xhat4.x = ((_FLOAT_PREC)xread4.x - mean) * invVariance;
        xhat4.y = ((_FLOAT_PREC)xread4.y - mean) * invVariance;
        xhat4.z = ((_FLOAT_PREC)xread4.z - mean) * invVariance;
        xhat4.w = ((_FLOAT_PREC)xread4.w - mean) * invVariance;
        db += (_FLOAT_PREC)dyRead4.x;
        db += (_FLOAT_PREC)dyRead4.y;
        db += (_FLOAT_PREC)dyRead4.z;
        db += (_FLOAT_PREC)dyRead4.w;
        ds = mad(xhat4.x, (_FLOAT_PREC)dyRead4.x, ds);
        ds = mad(xhat4.y, (_FLOAT_PREC)dyRead4.y, ds);
        ds = mad(xhat4.z, (_FLOAT_PREC)dyRead4.z, ds);
        ds = mad(xhat4.w, (_FLOAT_PREC)dyRead4.w, ds);
    }

#endif
    barrier(CLK_GLOBAL_MEM_FENCE);

// This reduction is around .5% of total
#ifndef __AMDGCN__
#if MIOPEN_USE_FP16 == 1
#if(MIO_BN_USESAVED == 1)
    __local float lcl_data[MIO_BN_LDS_SIZE];
#endif

    lcl_data[lid] = (float)ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_ds = (float)ds;
    regLDSreduce(&temp_ds, lcl_data, lid, (float)1.0);
    ds = (_FLOAT_PREC)temp_ds;
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[lid] = (float)db;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_db = (float)db;
    regLDSreduce(&temp_db, lcl_data, lid, (float)1.0);
    db = (_FLOAT_PREC)temp_db;
#else
#if(MIO_BN_USESAVED == 1)
    __local _FLOAT_PREC lcl_data[MIO_BN_LDS_SIZE];
#endif

    lcl_data[lid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&ds, lcl_data, lid, (_FLOAT_PREC)1.0);
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[lid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&db, lcl_data, lid, (_FLOAT_PREC)1.0);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    __local _FLOAT_PREC lcl_ds[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT_PREC lcl_db[MIO_BN_LDSGCN_SIZE];
    barrier(CLK_LOCAL_MEM_FENCE);
    dpp_interleaved_reduction(&ds, &db);

    if((lid % 64) == 63)
    {
        unsigned int ldsidx = lid >> 6;
        lcl_ds[ldsidx]      = ds;
        lcl_db[ldsidx]      = db;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = db = (_FLOAT_PREC)0.;

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        ds += lcl_ds[i];
        db += lcl_db[i];
    }
#endif

    pscale           = lcl_scale;
    _FLOAT_PREC tmp1 = 0.;
    _FLOAT_PREC tmp2 = 0.;
    _FLOAT_PREC tmp3 = pscale * invVariance * INHW;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid == 0)
    {
#if MIOPEN_USE_FP16 == 1
        *(dbias + grpid)  = (temp_db >= (float)MAX_VAL) ? MAX_VAL : db;
        *(dscale + grpid) = (temp_ds >= (float)MAX_VAL || temp_ds < 0) ? MAX_VAL : ds;
#else
        *(dbias + grpid)  = db;
        *(dscale + grpid) = ds;
#endif
    }

    _FLOAT_PREC vals[MIO_MAX_READ];
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
            dyvalue        = (_FLOAT_PREC)(*(dy_in + index));
            xhat = ((_FLOAT_PREC)(*(x_in + index)) - mean) * invVariance;
#if MIOPEN_USE_FP16 == 1
            float temp_tmp1 = mad((float)NHW, (float)dyvalue, -temp_db);
            float temp_tmp2 = -((float)xhat) * temp_ds;
            float temp_vals = (float)tmp3 * (temp_tmp2 + temp_tmp1);
            vals[j]         = (_FLOAT_PREC)temp_vals;
#else
            tmp1          = mad(NHW, dyvalue, -db);
            tmp2          = -xhat * ds;
            vals[j]       = tmp3 * (tmp2 + tmp1);
#endif
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for(unsigned int j = 0; j < MIO_MAX_READ; j++)
        {
            unsigned int l    = k + j;
            nidx              = l / MIO_BN_HW;
            hwidx             = l - (nidx * MIO_BN_HW);
            index             = nidx * MIO_BN_CHW + chwid + hwidx;
            *(dx_out + index) = (_FLOAT)vals[j];
        }
    }

#if(MIO_BN_REMOUT)
    unsigned int remkeyout = (MIO_MAX_READ * lid) + MIO_BN_LESSOUT;
    for(unsigned int j = 0; j < MIO_MAX_READ; j++)
    {
        unsigned int l = remkeyout + j;
        nidx           = l / MIO_BN_HW;
        hwidx          = l - (nidx * MIO_BN_HW);
        index          = nidx * MIO_BN_CHW + chwid + hwidx;
        if(index < MIO_BN_NCHW)
        {
            dyvalue = (_FLOAT_PREC)(*(dy_in + index));
            xhat    = ((_FLOAT_PREC)(*(x_in + index)) - mean) * invVariance;
            tmp1    = mad(NHW, dyvalue, -db);
            tmp2    = -xhat * ds;
            vals[j] = tmp3 * (tmp2 + tmp1);
        }
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
            *(dx_out + index) = (_FLOAT)vals[j];
        }
    }
#endif
}

#elif(MIO_BN_VARIANT == 2)

#if(MIO_BN_USESAVED == 0)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormBwdSpatialFinalMeanVariance(__global _FLOAT* __restrict meanvarbuff,
                                           _FLOAT INHW,
                                           double epsilon)
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
    __local float lcl_data[MIO_BN_NGRPS];
    lcl_data[lid]   = (float)mean;
    float temp_mean = (float)mean;
    barrier(CLK_LOCAL_MEM_FENCE);

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
#else
    temp_mean = (float)0.;
    for(unsigned int i = 0; i < MIO_BN_NGRPS; i++)
    {
        temp_mean += lcl_data[i];
    }

#endif
    mean                = (_FLOAT)temp_mean;
    lcl_data[lid]       = (float)variance;
    float temp_variance = (float)variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_NGRPS > 64)
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
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        temp_variance += lcl_data[i];
    }
#endif
    variance = (_FLOAT)temp_variance;
#else
    __local _FLOAT lcl_data[MIO_BN_NGRPS];
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
#else
    mean = (_FLOAT)0.;
    for(unsigned int i = 0; i < MIO_BN_NGRPS; i++)
    {
        mean += lcl_data[i];
    }

#endif
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_NGRPS > 64)
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
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        variance += lcl_data[i];
    }
#endif
#endif

#else // DPP below
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

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        variance += lcl_variance[i];
        mean += lcl_mean[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
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
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormBwdSpatialMeanVariance(const __global _FLOAT* __restrict in,
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

        for(unsigned int n = 0; n < MIO_BN_N; n++)
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

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        mean += lcl_mean[i];
        variance += lcl_variance[i];
    }

#else
#if MIOPEN_USE_FP16 == 1
    __local float lcl_data[MIO_BN_NGRPS];
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
    __local _FLOAT lcl_data[MIO_BN_NGRPS];
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

#endif // end USESAVED == 0

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormBwdSpatialDScaleDBias(const __global _FLOAT* x_in,
                                     const __global _FLOAT* dy_in,
                                     __global _FLOAT* buff
#if(MIO_BN_USESAVED == 1)

                                     ,
                                     const __global _FLOAT* savedMean,
                                     const __global _FLOAT* savedInvVariance
#endif
                                     )
{

    unsigned int xgid    = get_global_id(0);
    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;

    _FLOAT mean    = (_FLOAT)0.;
    _FLOAT invVar  = (_FLOAT)0.;
    _FLOAT elemStd = (_FLOAT)0.;
    _FLOAT xhat    = (_FLOAT)0.;
    _FLOAT dscale  = (_FLOAT)0.;
    _FLOAT dbias   = (_FLOAT)0.;

    __local _FLOAT lmean, livar;

    if(ylid == 0)
    {
#if(MIO_BN_USESAVED == 0)
        unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
        unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;
        lmean                       = *(buff + meanstashindex); // load stashed mean
        livar                       = *(buff + varstashindex);
#else  // NO SAVED
        lmean = *(savedMean + xgid);
        livar = *(savedInvVariance + xgid);
#endif // SAVED
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
        mean   = lmean;
        invVar = livar;

        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            dbias += *(dy_in + index);
            elemStd = *(x_in + index) - mean;
            xhat    = elemStd * invVar;
            dscale  = mad(xhat, dy_in[index], dscale);
        }
    }

// REDUCE over DS and DB

#ifdef __AMDGCN__
    unsigned int ldsidx = ylid >> 6;
    __local _FLOAT lcl_db[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT lcl_ds[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&dscale, &dbias);
    if((ylid % 64) == 63)
    {
        lcl_db[ldsidx] = dbias;
        lcl_ds[ldsidx] = dscale;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    dbias = dscale = 0.;

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        dbias += lcl_db[i];
        dscale += lcl_ds[i];
    }

#else
#if MIOPEN_USE_FP16 == 1
    __local float lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = (float)dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_dbias = (float)dbias;
    regLDSreduce(&dbias, lcl_data, ylid, 1);
    dbias = (_FLOAT)temp_dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[ylid] = (float)dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_dscale = (float)dscale;
    regLDSreduce(&dscale, lcl_data, ylid, 1);
    dscale = (_FLOAT)temp_dscale;
#else
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&dbias, lcl_data, ylid, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[ylid] = dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&dscale, lcl_data, ylid, 1);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    // end reduction-----------

    if(ylid == 0)
    {
        unsigned int betaindex  = cidx + ygrp_sz * ygrp_id + 6;
        unsigned int gammaindex = cidx + ygrp_sz * ygrp_id + 4;
        buff[gammaindex]        = dscale;
        buff[betaindex]         = dbias;
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormBwdSpatialFinalDScaleDBias(__global _FLOAT* buff,
                                          __global _FLOAT* delta_scale,
                                          __global _FLOAT* delta_bias)
{

    _FLOAT ds = (_FLOAT)0.;
    _FLOAT db = (_FLOAT)0.;

    unsigned int lid     = get_local_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    int cidx             = MIO_BN_HW * xgid;

    for(int gn = 0; gn < MIO_BN_NGRPS; gn++)
    {
        unsigned int offset = gn * ygrp_sz + lid;
        if(offset < yngrps)
        { // modify to span larger number of groups
            unsigned int gammaindex = cidx + ygrp_sz * offset + 4;
            unsigned int betaindex  = cidx + ygrp_sz * offset + 6;
            ds += *(buff + gammaindex);
            db += *(buff + betaindex);
        }
    }

#ifndef __AMDGCN__
#if MIOPEN_USE_FP16 == 1
    __local float lcl_data[MIO_BN_NGRPS];
    lcl_data[lid] = (float)ds;
    float temp_ds = (float)ds;
    barrier(CLK_LOCAL_MEM_FENCE);
#if(MIO_BN_NGRPS > 256)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&temp_ds, lcl_data, lid, 1.);
#elif(MIO_BN_NGRPS <= 256)
    regLDSreduce(&temp_ds, lcl_data, lid, 1.);
    commitID = 0;
#else
    temp_ds = (float)0.;

    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        temp_ds += lcl_data[i];
    }

#endif
    ds            = (_FLOAT)temp_ds;
    lcl_data[lid] = (float)db;
    float temp_db = (float)db;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_NGRPS > 256)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&temp_db, lcl_data, lid, 1.);
#elif(MIO_BN_NGRPS <= 256)
    regLDSreduce(&temp_db, lcl_data, lid, 1.);
#else //(MIO_BN_NGRPS <= 16)
    temp_db = (float)0.;

    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        temp_db += lcl_data[i];
    }
#endif
    db = (_FLOAT)temp_db;
#else
    __local _FLOAT lcl_data[MIO_BN_NGRPS];
    lcl_data[lid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
#if(MIO_BN_NGRPS > 256)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&ds, lcl_data, lid, 1.);
#elif(MIO_BN_NGRPS <= 256)
    regLDSreduce(&ds, lcl_data, lid, 1.);
    commitID = 0;
#else
    ds = (_FLOAT)0.;

    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        ds += lcl_data[i];
    }

#endif

    lcl_data[lid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_NGRPS > 256)
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&db, lcl_data, lid, 1.);
#elif(MIO_BN_NGRPS <= 256)
    regLDSreduce(&db, lcl_data, lid, 1.);
#else //(MIO_BN_NGRPS <= 16)
    db = (_FLOAT)0.;

    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        db += lcl_data[i];
    }
#endif
#endif
#else // DPP below
    unsigned int ldsidx = lid >> 6;
    __local _FLOAT lcl_ds[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT lcl_db[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&ds, &db);
    if((lid % 64) == 63)
    {
        lcl_ds[ldsidx] = ds;
        lcl_db[ldsidx] = db;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    ds = db = 0.;

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        ds += lcl_ds[i];
        db += lcl_db[i];
    }
#endif

    if(ygid == 0)
    {
        delta_scale[xgid] = ds;
        delta_bias[xgid]  = db;
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormBwdSpatialDX(const __global _FLOAT* x_in,
                            const __global _FLOAT* dy_in,
                            __global _FLOAT* dx_out,
                            const __global _FLOAT* bnScale,
                            __global _FLOAT* delta_scale,
                            __global _FLOAT* delta_bias,
#if(MIO_BN_USESAVED == 1)
                            const __global _FLOAT* savedMean,
                            const __global _FLOAT* savedInvVariance,
#endif
                            _FLOAT INHW)
{

    int xgid = get_global_id(0);
    int ygid = get_global_id(1);
    int cidx = MIO_BN_HW * xgid;
    unsigned int index;
    _FLOAT mean, invVar;
    _FLOAT elemStd, xhat;
    _FLOAT scale, dscale, dbias;
    _FLOAT tmp1, tmp2, tmp3;
    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    local _FLOAT lscale, ldscale, ldbias, lmean, livar;

    if(get_local_id(1) == 0)
    {

#if(MIO_BN_USESAVED == 0)
        int ygrp_id                 = get_group_id(1);
        int ygrp_sz                 = get_local_size(1);
        unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
        unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;
        lmean                       = *(dx_out + meanstashindex); // load stashed mean
        livar                       = *(dx_out + varstashindex);
#else  // SAVED
        lmean = *(savedMean + xgid);
        livar = *(savedInvVariance + xgid);
#endif // SAVED
        lscale                      = *(bnScale + xgid);
        ldscale                     = *(delta_scale + xgid);
        ldbias                      = *(delta_bias + xgid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //________________________________________________
    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ygid < MIO_BN_HW)
    {

        mean   = lmean;
        invVar = livar;
        scale  = lscale;
        dscale = ldscale;
        dbias  = ldbias;

        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index         = n * MIO_BN_CHW + cidx + ygid;
            elemStd       = *(x_in + index) - mean; // (x_i - mean)
            xhat          = elemStd * invVar;       // recalculating this again...
            tmp1          = mad(NHW, *(dy_in + index), -dbias);
            tmp2          = -xhat * dscale;
            tmp3          = scale * invVar * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1);
        }
    }
}

//============================================================

#elif(MIO_BN_VARIANT == 3)

//=============== SINGLE WORKGROUP PER CHANNEL

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormBwdSpatial(const __global _FLOAT* __restrict x_in,
                          const __global _FLOAT* __restrict dy_in,
                          __global _FLOAT* __restrict dx_out,
                          const __global _FLOAT_PREC* bnScale,
                          __global _FLOAT_PREC* __restrict dscale,
                          __global _FLOAT_PREC* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                          double epsilon,
#elif(MIO_BN_USESAVED == 1)
                          const __global _FLOAT_PREC* savedMean,
                          const __global _FLOAT_PREC* savedInvVariance,
#endif
                          _FLOAT_PREC INHW)
{

    // SPATIAL
    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
#if(MIO_BN_USESAVED == 0)
    _FLOAT_PREC variance    = (_FLOAT_PREC)0.;
#endif
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pscale      = (_FLOAT_PREC)0.;
    _FLOAT_PREC ds          = (_FLOAT_PREC)0.;
    _FLOAT_PREC db          = (_FLOAT_PREC)0.;
#if(MIO_BN_N < MIO_BN_MAXN)
    _FLOAT_PREC batchvalues[MIO_BN_N];
    _FLOAT_PREC dyvalues[MIO_BN_N];
#endif

    unsigned int lid   = get_local_id(0);
    unsigned int grpid = get_group_id(0);
    unsigned int index;
    unsigned int cidx = grpid * MIO_BN_HW;
    _FLOAT_PREC tmp1, tmp2, tmp3;

#if(MIO_BN_USESAVED == 1)
    __local _FLOAT_PREC lmean, lvar;
#endif

    _FLOAT_PREC NHW = (_FLOAT_PREC)MIO_BN_NHW;
    __local _FLOAT_PREC lcl_scale;

    if(lid == 0)
    {
        lcl_scale = *(bnScale + grpid);

#if(MIO_BN_USESAVED == 1)

        lmean = *(savedMean + grpid);
        lvar  = *(savedInvVariance + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean        = lmean;
    invVariance = lvar;

#else // recalc mean and variance

    } // end if(!lid)

    if(lid < MIO_BN_HW)
    {
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index                  = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            mean += batchvalues[n] = (_FLOAT_PREC)(*(x_in + index));
            variance               = mad(batchvalues[n], batchvalues[n], variance);
#else
            _FLOAT_PREC in = (_FLOAT_PREC)(*(x_in + index));
            mean += in;
            variance = mad(in, in, variance);
#endif
        }
    }
    else
    {
        mean     = (_FLOAT_PREC)0.;
        variance = (_FLOAT_PREC)0.;
    }

// REDUCE MEAN AND VARIANCE -----------------------
#ifndef __AMDGCN__
#if MIOPEN_USE_FP16 == 1
    local float lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[lid] = (float)mean;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
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

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
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

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&mean, lcl_data, lid, (_FLOAT_PREC)INHW);

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&variance, lcl_data, lid, (_FLOAT_PREC)INHW);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

#else

    __local _FLOAT_PREC lcl_mean[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT_PREC lcl_variance[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&mean, &variance);
    unsigned int ldsidx1 = lid >> 6;
    if((lid % 64) == 63)
    {
        lcl_mean[ldsidx1]     = mean;
        lcl_variance[ldsidx1] = variance;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = variance = (_FLOAT_PREC)0.;

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        mean += lcl_mean[i];
        variance += lcl_variance[i];
    }
    mean *= (_FLOAT_PREC)INHW;
    variance *= (_FLOAT_PREC)INHW;
#endif
    // REDUCTION COMPLETE ---------------------------
    barrier(CLK_LOCAL_MEM_FENCE);
    variance    = mad(-mean, mean, variance);
    invVariance = rsqrt(variance + epsilon);

// RECALC of MEAN and VARIANCE complete
//===============================================
#endif

    if(lid < MIO_BN_HW)
    {
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index             = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            db += dyvalues[n] = (_FLOAT_PREC)(*(dy_in + index));

#if(MIO_BN_USESAVED == 1)
            batchvalues[n]    = ((_FLOAT_PREC)(*(x_in + index)) - mean) * invVariance;
#else
            batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif // batchvalues is now xhat

            ds = mad(batchvalues[n], dyvalues[n], ds);
#else  // maxn
            db += (_FLOAT_PREC)(*(dy_in + index));
            _FLOAT_PREC xhat = (((_FLOAT_PREC)(*(x_in + index)) - mean) * invVariance);
            ds = mad(xhat, (_FLOAT_PREC)(*(dy_in + index)), ds);
#endif
        }
    }
    else
    {
        db = (_FLOAT_PREC)0.;
        ds = (_FLOAT_PREC)0.;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#ifndef __AMDGCN__
#if MIOPEN_USE_FP16 == 1
#if(MIO_BN_USESAVED == 1)
    __local float lcl_data[MIO_BN_LDS_SIZE];
#endif

    lcl_data[lid] = (float)ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_ds = (float)ds;
    regLDSreduce(&temp_ds, lcl_data, lid, (float)1.0);
    ds = (_FLOAT_PREC)temp_ds;
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[lid] = (float)db;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float temp_db = (float)db;
    regLDSreduce(&temp_db, lcl_data, lid, (float)1.0);
    db = (_FLOAT_PREC)temp_db;
#else
#if(MIO_BN_USESAVED == 1)
    __local _FLOAT_PREC lcl_data[MIO_BN_LDS_SIZE];
#endif

    lcl_data[lid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&ds, lcl_data, lid, (_FLOAT_PREC)1.0);
    barrier(CLK_LOCAL_MEM_FENCE);

    lcl_data[lid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&db, lcl_data, lid, (_FLOAT_PREC)1.0);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    __local _FLOAT_PREC lcl_ds[MIO_BN_LDSGCN_SIZE];
    __local _FLOAT_PREC lcl_db[MIO_BN_LDSGCN_SIZE];

    dpp_interleaved_reduction(&ds, &db);

    unsigned int ldsidx2 = lid >> 6;
    if((lid % 64) == 63)
    {
        lcl_ds[ldsidx2] = ds;
        lcl_db[ldsidx2] = db;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = db = (_FLOAT_PREC)0.;

    for(unsigned int i = 0; i < MIO_BN_LDSGCN_SIZE; i++)
    {
        ds += lcl_ds[i];
        db += lcl_db[i];
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(lid < MIO_BN_HW)
    {
        pscale = lcl_scale;
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index         = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            tmp1          = mad(NHW, dyvalues[n], -db);
            tmp2          = -(batchvalues[n]) * ds;
#else
            tmp1 = mad(NHW, (_FLOAT_PREC)(*(dy_in + index)), -db);
            tmp2 = -((_FLOAT_PREC)(*(x_in + index)) - mean) * invVariance * ds;
#endif
            tmp3          = (pscale * invVariance) * INHW;
            dx_out[index] = (_FLOAT)(tmp3 * (tmp2 + tmp1));
        }
    }
    if(lid == 0)
    {
        dbias[grpid]  = db;
        dscale[grpid] = ds;
    }

} // end spatial

#endif // END VARIANTS

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
