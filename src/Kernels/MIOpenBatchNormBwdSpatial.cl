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

#define UNUSED __attribute__((__unused__))

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
BatchNormBwdSpatialSavedSingleDX(const __global _FLOAT* __restrict x_in,
                                 const __global _FLOAT* __restrict dy_in,
                                 __global _FLOAT* __restrict dx_out,
                                 const __global _FLOAT* bnScale,
                                 __global _FLOAT* __restrict dscale,
                                 __global _FLOAT* __restrict dbias,
                                 const __global _FLOAT* savedMean,
                                 const __global _FLOAT* savedInvVariance,
                                 float INHW)
{

    // SPATIAL
    _FLOAT mean   = 0.;
    _FLOAT invVar = 0.;
    _FLOAT xhat   = 0.;
    _FLOAT pscale = 0.;
    _FLOAT ds     = 0.;
    _FLOAT db     = 0.;
    _FLOAT elemStd;
    _FLOAT tmp1, tmp2, tmp3;

    __local _FLOAT lbns, lmean, lvar;

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    if(ylid == 0)
    {
        lbns  = bnScale[xgid];
        lmean = savedMean[xgid];
        lvar  = savedInvVariance[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    mean   = lmean;
    invVar = lvar;

    if(ygid < MIO_BN_N)
    {
        _FLOAT de = 0.;
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index = ygid * MIO_BN_CHW + cidx + hw;
            de    = dy_in[index];
            db += de;
            elemStd = x_in[index] - mean; // (x_i - mean)
            xhat    = elemStd * invVar;
            ds      = mad(xhat, de, ds);
        }
    }

#ifdef __AMDGCN__
    __local _FLOAT ldb;
    __local _FLOAT lds;

#if(MIO_BN_GRP1 > 64)

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    _FLOAT tmp     = 0.;
    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    db = lcl_data[ylid];
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_data[ylid];
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;
    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#elif(MIO_BN_GRP1 > 16)

    _FLOAT tmp = 0.;
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;
    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#else

    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x101, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x102, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x104, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x108, 15, 15, 0));

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x101, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x102, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x104, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x108, 15, 15, 0));
    ds *= INHW;

    if(ylid == 0)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#endif

#else

    __local _FLOAT lcl_ds[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_db[MIO_BN_LDS_SIZE];

    lcl_ds[ylid] = ds;
    lcl_db[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_GRP1 > 16)

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
    {
        ReduceKernel(lcl_ds, 1, ylid, 4);
        ReduceKernel(lcl_db, 1, ylid, 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
    {
        ReduceKernel(lcl_ds, 4, ylid, 16);
        ReduceKernel(lcl_db, 4, ylid, 16);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
    {
        ReduceKernel(lcl_ds, 16, ylid, MIO_BN_LDS_SIZE);
        ReduceKernel(lcl_db, 16, ylid, MIO_BN_LDS_SIZE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_ds[0];
    db = lcl_db[0];

#else

    lcl_ds[ylid] = ds;
    lcl_db[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        db += lcl_db[i];
        ds += lcl_ds[i];
    }

#endif

    ds *= INHW;

#endif

    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ygid < MIO_BN_N)
    {

#ifdef __AMDGCN__
        db = ldb;
        ds = lds;
#endif

        pscale = lbns;
#pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index         = ygid * MIO_BN_CHW + cidx + hw;
            elemStd       = x_in[index] - mean; // (x_i - mean)
            xhat          = elemStd * invVar;   // recalculating this again...
            tmp1          = mad(NHW, dy_in[index], -db);
            tmp2          = -xhat * ds;
            tmp3          = (pscale * invVar) * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1); // DEBUG
        }
    }
    if(ygid == 0)
    {
        dbias[xgid]  = db;
        dscale[xgid] = ds;
    }

} // end spatial

// Recalc everything
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSingleDX(const __global _FLOAT* __restrict x_in,
                            const __global _FLOAT* __restrict dy_in,
                            __global _FLOAT* __restrict dx_out,
                            const __global _FLOAT* bnScale,
                            __global _FLOAT* __restrict dscale,
                            __global _FLOAT* __restrict dbias,
                            double epsilon,
                            float INHW)
{

    // SPATIAL
    _FLOAT mean     = 0.;
    _FLOAT variance = 0.;
    _FLOAT invVar   = 0.;
    _FLOAT xhat     = 0.;
    _FLOAT pscale   = 0.;
    _FLOAT ds       = 0.;
    _FLOAT db       = 0.;
    _FLOAT elemStd  = 0.;

    __local _FLOAT lbns;

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    _FLOAT tmp1, tmp2, tmp3;

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    if(ylid == 0)
    {
        lbns = bnScale[xgid];
    }

    if(ygid < MIO_BN_N)
    {
#pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index = ygid * MIO_BN_CHW + cidx + hw;
            mean += x_in[index];
        }
    }

#ifdef __AMDGCN__
    __local _FLOAT lds, ldb;
    __local _FLOAT lmean, lvariance;
#if(MIO_BN_GRP1 > 64)
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
        lmean = mean * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lmean;

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
    if(ylid == 63)
    {
        lmean = mean * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lmean;
#else
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x108, 15, 15, 0));
    if(ylid == 0)
    {
        lmean = mean * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lmean;
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
    mean = lcl_data[0] * INHW;
#else
    mean = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        mean += lcl_data[i];
    }
    mean *= INHW;
#endif

#endif

    if(ygid < MIO_BN_N)
    {
#pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index   = ygid * MIO_BN_CHW + cidx + hw;
            elemStd = (x_in[index] - mean);
            variance += elemStd * elemStd;
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_GRP1 > 64)

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
        lvariance = variance * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvariance;

#elif(MIO_BN_GRP1 > 16)
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
        lvariance = variance * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvariance;
#else
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x108, 15, 15, 0));
    if(ylid == 0)
    {
        lvariance = variance * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvariance;
#endif

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
    invVar = rsqrt(variance + epsilon);

    if(ygid < MIO_BN_HW)
    {
        _FLOAT de = 0.;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            de    = dy_in[index];
            db += de;
            elemStd = x_in[index] - mean; // (x_i - mean)
            xhat    = elemStd * invVar;
            ds      = mad(xhat, de, ds);
        }
    }

#ifdef __AMDGCN__
#if(MIO_BN_GRP1 > 64)

    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    db = lcl_data[ylid];
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_data[ylid];
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;
    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#elif(MIO_BN_GRP1 > 16)

    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;
    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x101, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x102, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x104, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x108, 15, 15, 0));

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x101, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x102, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x104, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x108, 15, 15, 0));
    ds *= INHW;

    if(ylid == 0)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
#else

    __local _FLOAT lcl_db[MIO_BN_LDS_SIZE];

    lcl_data[ylid] = ds;
    lcl_db[ylid]   = db;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_GRP1 > 16)

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
    {
        ReduceKernel(lcl_data, 1, ylid, 4);
        ReduceKernel(lcl_db, 1, ylid, 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
    {
        ReduceKernel(lcl_data, 4, ylid, 16);
        ReduceKernel(lcl_db, 4, ylid, 16);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
    {
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
        ReduceKernel(lcl_db, 16, ylid, MIO_BN_LDS_SIZE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_data[0];
    db = lcl_db[0];

#else

    lcl_data[ylid] = ds;
    lcl_db[ylid]   = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        db += lcl_db[i];
        ds += lcl_data[i];
    }
#endif

// ds *= INHW;

#endif

    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ygid < MIO_BN_N)
    {
#ifdef __AMDGCN__
        db = ldb;
        ds = lds;
#endif
        pscale = lbns;
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index         = ygid * MIO_BN_CHW + cidx + hw;
            elemStd       = x_in[index] - mean; // (x_i - mean)
            xhat          = elemStd * invVar;   // recalculating this again...
            tmp1          = mad(NHW, dy_in[index], -db);
            tmp2          = -xhat * ds;
            tmp3          = (pscale * invVar) * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1);
        }
    }
    if(ygid == 0)
    {
        dbias[xgid]  = db;
        dscale[xgid] = ds;
    }

} // end spatial

#elif(MIO_BN_VARIANT == 1)

//=============== SINGLE WORKGROUP PER CHANNEL

// Recalc everything
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSingleLDSDX(const __global _FLOAT* __restrict x_in,
                               const __global _FLOAT* __restrict dy_in,
                               __global _FLOAT* __restrict dx_out,
                               const __global _FLOAT* bnScale,
                               __global _FLOAT* __restrict dscale,
                               __global _FLOAT* __restrict dbias,
                               double epsilon,
                               float INHW)
{

    // SPATIAL
    _FLOAT mean     = 0.;
    _FLOAT variance = 0.;
    _FLOAT invVar   = 0.;
    _FLOAT xhat     = 0.;
    _FLOAT pscale   = 0.;
    _FLOAT elemStd  = 0.;
    _FLOAT ds       = 0.;
    _FLOAT db       = 0.;

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    _FLOAT tmp1, tmp2, tmp3;

    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_scale;

    if(ylid == 0)
        lcl_scale = bnScale[xgid];

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index               = n * MIO_BN_CHW + cidx + ygid;
            lcl_indata[n][ylid] = x_in[index];
            mean += lcl_indata[n][ylid];
        }
    }

#ifdef __AMDGCN__

    __local _FLOAT lmean, lvariance, lds, ldb;

#if(MIO_BN_GRP1 > 64)
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
        lmean = mean * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lmean;

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
    if(ylid == 63)
    {
        lmean = mean * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lmean;
#else
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x108, 15, 15, 0));
    if(ylid == 0)
    {
        lmean = mean * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lmean;
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
    mean = lcl_data[0] * INHW;
#else

    mean = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        mean += lcl_data[i];
    }
    mean *= INHW;
#endif
#endif

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index    = n * MIO_BN_CHW + cidx + ygid;
            elemStd  = (lcl_indata[n][ylid] - mean);
            variance = mad(elemStd, elemStd, variance);
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_GRP1 > 64)

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
        lvariance = variance * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvariance;

#elif(MIO_BN_GRP1 > 16)

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
        lvariance = variance * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvariance;
#else
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x108, 15, 15, 0));
    if(ylid == 0)
    {
        lvariance = variance * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvariance;
#endif

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
    invVar = rsqrt(variance + epsilon);
    if(ygid < MIO_BN_HW)
    {
        _FLOAT de = 0.;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            de    = dy_in[index];
            db += de;
            elemStd = x_in[index] - mean; // (x_i - mean)
            xhat    = elemStd * invVar;
            ds      = mad(xhat, de, ds);
        }
    }

#ifdef __AMDGCN__
#if(MIO_BN_GRP1 > 64)

    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    db = lcl_data[ylid];
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_data[ylid];
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;
    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#elif(MIO_BN_GRP1 > 16)

    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;
    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x101, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x102, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x104, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x108, 15, 15, 0));

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x101, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x102, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x104, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x108, 15, 15, 0));
    ds *= INHW;

    if(ylid == 0)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#else

    __local _FLOAT lcl_db[MIO_BN_LDS_SIZE];

    lcl_data[ylid] = ds;
    lcl_db[ylid]   = db;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_GRP1 > 16)

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
    {
        ReduceKernel(lcl_data, 1, ylid, 4);
        ReduceKernel(lcl_db, 1, ylid, 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
    {
        ReduceKernel(lcl_data, 4, ylid, 16);
        ReduceKernel(lcl_db, 4, ylid, 16);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
    {
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
        ReduceKernel(lcl_db, 16, ylid, MIO_BN_LDS_SIZE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_data[0];
    db = lcl_db[0];

#else

    db = ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        db += lcl_db[i];
        ds += lcl_data[i];
    }
#endif
    ds *= INHW;
#endif

    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ygid < MIO_BN_HW)
    {
#ifdef __AMDGCN__
        db = ldb;
        ds = lds;
#endif

        pscale = lcl_scale;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index         = n * MIO_BN_CHW + cidx + ygid;
            elemStd       = lcl_indata[n][ylid] - mean; // (x_i - mean)
            xhat          = elemStd * invVar;           // recalculating this again...
            tmp1          = mad(NHW, dy_in[index], -db);
            tmp2          = -xhat * ds;
            tmp3          = (pscale * invVar) * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1);
        }
    }
    if(ygid == 0)
    {
        dbias[xgid]  = db;
        dscale[xgid] = ds;
    }

} // end spatial

#elif(MIO_BN_VARIANT == 2)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSavedSingleLDSDX(const __global _FLOAT* __restrict x_in,
                                    const __global _FLOAT* __restrict dy_in,
                                    __global _FLOAT* __restrict dx_out,
                                    const __global _FLOAT* bnScale,
                                    __global _FLOAT* __restrict dscale,
                                    __global _FLOAT* __restrict dbias,
                                    const __global _FLOAT* savedMean,
                                    const __global _FLOAT* savedInvVariance,
                                    float INHW)
{

    // SPATIAL
    __private _FLOAT mean   = 0.;
    __private _FLOAT invVar = 0.;
    __private _FLOAT xhat   = 0.;
    __private _FLOAT pscale = 0.;
    __private _FLOAT ds     = 0.;
    __private _FLOAT db     = 0.;
    __private _FLOAT elemStd;
    _FLOAT tmp1, tmp2, tmp3;

    __local _FLOAT lcl_indata[MIO_BN_LDS_NSIZE][MIO_BN_LDS_HWSIZE];
    __local _FLOAT lcl_scale, lcl_mean, lcl_ivar;

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    unsigned int idx  = cidx + ygid;

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    if(ylid == 0)
    {
        lcl_mean  = savedMean[xgid];
        lcl_ivar  = savedInvVariance[xgid];
        lcl_scale = bnScale[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    invVar = lcl_ivar;
    mean   = lcl_mean;

    if(ygid < MIO_BN_HW)
    {
        _FLOAT de = 0.;
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + idx;
            de    = dy_in[index];
            db += de;
            lcl_indata[n][ylid] = elemStd = x_in[index] - mean;
            xhat                          = elemStd * invVar;
            ds                            = mad(xhat, de, ds);
        }
    }

#ifdef __AMDGCN__

    __local _FLOAT ldb, lds;
#if(MIO_BN_GRP1 > 64)

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    _FLOAT tmp = 0;

    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    db = lcl_data[ylid];
    barrier(CLK_LOCAL_MEM_FENCE);
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_data[ylid];
    // barrier(CLK_LOCAL_MEM_FENCE);
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;

    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#elif(MIO_BN_GRP1 > 16)

    _FLOAT tmp = 0.;
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;

    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x101, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x102, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x104, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x108, 15, 15, 0));

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x101, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x102, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x104, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x108, 15, 15, 0));
    ds *= INHW;

    if(ylid == 0)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#else

    __local _FLOAT lcl_ds[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_db[MIO_BN_LDS_SIZE];

    lcl_ds[ylid] = ds;
    lcl_db[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_GRP1 > 16)

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
    {
        ReduceKernel(lcl_ds, 1, ylid, 4);
        ReduceKernel(lcl_db, 1, ylid, 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
    {
        ReduceKernel(lcl_ds, 4, ylid, 16);
        ReduceKernel(lcl_db, 4, ylid, 16);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
    {
        ReduceKernel(lcl_ds, 16, ylid, MIO_BN_LDS_SIZE);
        ReduceKernel(lcl_db, 16, ylid, MIO_BN_LDS_SIZE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_ds[0];
    db = lcl_db[0];

#else

    lcl_ds[ylid] = ds;
    lcl_db[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        db += lcl_db[i];
        ds += lcl_ds[i];
    }

#endif

    ds *= INHW;

#endif

    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack

    if(ylid < MIO_BN_HW)
    {
        pscale = lcl_scale;
#ifdef __AMDGCN__
        db     = ldb;
        ds     = lds;
#endif
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index         = n * MIO_BN_CHW + idx;
            xhat          = lcl_indata[n][ylid] * invVar; // recalculating this again...
            tmp1          = mad(NHW, dy_in[index], -db);
            tmp2          = -xhat * ds;
            tmp3          = (pscale * invVar) * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1); // DEBUG
        }
    }
    if(ygid == 0)
    {
        dbias[xgid]  = db;
        dscale[xgid] = ds;
    }
} // end spatial

#elif(MIO_BN_VARIANT == 3)

// Recalc everything
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSingleDX(const __global _FLOAT* __restrict x_in,
                            const __global _FLOAT* __restrict dy_in,
                            __global _FLOAT* __restrict dx_out,
                            const __global _FLOAT* bnScale,
                            __global _FLOAT* __restrict dscale,
                            __global _FLOAT* __restrict dbias,
                            double epsilon,
                            float INHW)
{

    // SPATIAL
    _FLOAT mean     = 0.;
    _FLOAT variance = 0.;
    _FLOAT invVar   = 0.;
    _FLOAT xhat     = 0.;
    _FLOAT pscale   = 0.;
    _FLOAT ds       = 0.;
    _FLOAT db       = 0.;
    _FLOAT elemStd  = 0.;
    _FLOAT tmp1, tmp2, tmp3;

    local _FLOAT lbns;

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    _FLOAT NHW        = (_FLOAT)MIO_BN_NHW;

    if(ylid == 0)
        lbns = bnScale[xgid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            mean += x_in[index];
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_GRP1 > 64)

    __local _FLOAT lds, ldb, lmean, lvariance, __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

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
        lmean = mean * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lmean;

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
    if(ylid == 63)
    {
        lmean = mean * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lmean;
#else
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x108, 15, 15, 0));
    if(ylid == 0)
    {
        lmean = mean * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = lmean;
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
    mean = lcl_data[0] * INHW;
#else
    mean = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        mean += lcl_data[i];
    }
    mean *= INHW;
#endif

#endif

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index   = n * MIO_BN_CHW + cidx + ygid;
            elemStd = (x_in[index] - mean);
            variance += elemStd * elemStd;
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_GRP1 > 64)

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
        lvariance = variance * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvariance;

#elif(MIO_BN_GRP1 > 16)
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
        lvariance = variance * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvariance;
#else
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x108, 15, 15, 0));
    if(ylid == 0)
    {
        lvariance = variance * INHW;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = lvariance;
#endif

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
    invVar = rsqrt(variance + epsilon);

    if(ygid < MIO_BN_HW)
    {
        _FLOAT de = 0.;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            de    = dy_in[index];
            db += de;
            elemStd = x_in[index] - mean; // (x_i - mean)
            xhat    = elemStd * invVar;
            ds      = mad(xhat, de, ds);
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_GRP1 > 64)

    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    db = lcl_data[ylid];
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_data[ylid];
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;

    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#elif(MIO_BN_GRP1 > 16)

    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;

    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x101, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x102, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x104, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x108, 15, 15, 0));

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x101, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x102, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x104, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x108, 15, 15, 0));
    ds *= INHW;

    if(ylid == 0)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

#else

    __local _FLOAT lcl_db[MIO_BN_LDS_SIZE];

    lcl_data[ylid] = ds;
    lcl_db[ylid]   = db;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_GRP1 > 16)

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
    {
        ReduceKernel(lcl_data, 1, ylid, 4);
        ReduceKernel(lcl_db, 1, ylid, 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
    {
        ReduceKernel(lcl_data, 4, ylid, 16);
        ReduceKernel(lcl_db, 4, ylid, 16);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
    {
        ReduceKernel(lcl_data, 16, ylid, MIO_BN_LDS_SIZE);
        ReduceKernel(lcl_db, 16, ylid, MIO_BN_LDS_SIZE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_data[0];
    db = lcl_db[0];

#else

    lcl_data[ylid] = ds;
    lcl_db[ylid]   = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        db += lcl_db[i];
        ds += lcl_data[i];
    }

#endif

    ds *= INHW;

#endif

    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ygid < MIO_BN_HW)
    {
        pscale = lbns;
#ifdef __AMDGCN__
        db     = ldb;
        ds     = lds;
#endif

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index         = n * MIO_BN_CHW + cidx + ygid;
            elemStd       = x_in[index] - mean; // (x_i - mean)
            xhat          = elemStd * invVar;   // recalculating this again...
            tmp1          = mad(NHW, dy_in[index], -db);
            tmp2          = -xhat * ds;
            tmp3          = (pscale * invVar) * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1);
        }
    }
    if(ygid == 0)
    {
        dbias[xgid]  = db;
        dscale[xgid] = ds;
    }

} // end spatial

#elif(MIO_BN_VARIANT == 4)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSavedSingleDX(const __global _FLOAT* __restrict x_in,
                                 const __global _FLOAT* __restrict dy_in,
                                 __global _FLOAT* __restrict dx_out,
                                 const __global _FLOAT* bnScale,
                                 __global _FLOAT* __restrict dscale,
                                 __global _FLOAT* __restrict dbias,
                                 const __global _FLOAT* savedMean,
                                 const __global _FLOAT* savedInvVariance,
                                 float INHW)
{

    // SPATIAL
    _FLOAT mean   = 0.;
    _FLOAT invVar = 0.;
    _FLOAT xhat   = 0.;
    _FLOAT pscale = 0.;
    _FLOAT ds     = 0.;
    _FLOAT db     = 0.;
    _FLOAT tmp1, tmp2, tmp3;

    __local _FLOAT lcl_scale, lcl_mean, lcl_ivar;

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    unsigned int idx  = cidx + ygid;

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    if(ylid == 0)
    {
        lcl_mean  = savedMean[xgid];
        lcl_ivar  = savedInvVariance[xgid];
        lcl_scale = bnScale[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    invVar = lcl_ivar;
    mean   = lcl_mean;

    if(ygid < MIO_BN_HW)
    {
        _FLOAT de = 0.;
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + idx;
            de    = dy_in[index];
            db += de;
            xhat = (x_in[index] - mean) * invVar;
            ds   = mad(xhat, de, ds);
        }
    }

#ifdef __AMDGCN__
    __local _FLOAT lds, ldb;
#if(MIO_BN_GRP1 > 64)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    _FLOAT tmp = 0.;

    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    db = lcl_data[ylid];
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_data[ylid];
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;

    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#elif(MIO_BN_GRP1 > 16)

    _FLOAT tmp = 0.;
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x111, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x112, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x114, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x142, 15, 15, 0));
    ds += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x143, 15, 15, 0));
    ds += tmp;
    ds *= INHW;

    if(ylid == 63)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x101, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x102, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x104, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x108, 15, 15, 0));

    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x101, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x102, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x104, 15, 15, 0));
    ds += as_float(__builtin_amdgcn_mov_dpp(as_int(ds), 0x108, 15, 15, 0));
    ds *= INHW;

    if(ylid == 0)
    {
        ldb = db;
        lds = ds;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#endif

#else
    __local _FLOAT lcl_ds[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_db[MIO_BN_LDS_SIZE];
    lcl_ds[ylid] = ds;
    lcl_db[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

#if(MIO_BN_GRP1 > 16)

    if(ylid < (MIO_BN_LDS_SIZE >> 2))
    {
        ReduceKernel(lcl_ds, 1, ylid, 4);
        ReduceKernel(lcl_db, 1, ylid, 4);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < (MIO_BN_LDS_SIZE >> 4))
    {
        ReduceKernel(lcl_ds, 4, ylid, 16);
        ReduceKernel(lcl_db, 4, ylid, 16);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid == 0)
    {
        ReduceKernel(lcl_ds, 16, ylid, MIO_BN_LDS_SIZE);
        ReduceKernel(lcl_db, 16, ylid, MIO_BN_LDS_SIZE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = lcl_ds[0];
    db = lcl_db[0];

#else

    lcl_ds[ylid] = ds;
    lcl_db[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_GRP1; i++)
    {
        db += lcl_db[i];
        ds += lcl_ds[i];
    }

#endif

    ds *= INHW;

#endif

    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ylid < MIO_BN_HW)
    {
        pscale = lcl_scale;
#ifdef __AMDGCN__
        ds     = lds;
        db     = ldb;
#endif
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index         = n * MIO_BN_CHW + idx;
            xhat          = (x_in[index] - mean) * invVar; // recalculating this again...
            tmp1          = mad(NHW, dy_in[index], -db);
            tmp2          = -xhat * ds;
            tmp3          = (pscale * invVar) * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1); // DEBUG
        }
    }
    if(ygid == 0)
    {
        dbias[xgid]  = db;
        dscale[xgid] = ds;
    }

} // end spatial

#elif(MIO_BN_VARIANT == 5)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialMean(const __global _FLOAT* __restrict in, __global _FLOAT* __restrict meanbuff)
{

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx      = xgid * MIO_BN_HW;
    unsigned int meanindex = cidx + ygrp_sz * ygrp_id; // making assumption of n=0 here
    _FLOAT mean            = 0.;

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
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

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialFinalMean(__global _FLOAT* __restrict meanvarbuff)
{

    __private _FLOAT mean = 0.;

    unsigned int ylid     = get_local_id(1);
    unsigned int ygrp_id  = get_group_id(1);
    unsigned int xgid     = get_global_id(0);
    unsigned int ygrp_sz  = get_local_size(1);
    unsigned int yngrps   = get_num_groups(1);
    unsigned int cidx     = xgid * MIO_BN_HW;
    unsigned int commitID = 0;

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;
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
    mean /= NHW;
    commitID = 63;

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
    mean /= NHW;
    commitID = 63;

#else
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x101, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x102, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x104, 15, 15, 0));
    mean += as_float(__builtin_amdgcn_mov_dpp(as_int(mean), 0x108, 15, 15, 0));
    mean /= NHW;
    commitID = 0;

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
    mean     = lcl_data[0] / NHW;
    commitID = 0;

#else

    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        mean += lcl_data[i];
    }
    mean /= NHW;

#endif

#endif

    if(ylid == commitID)
    {
        unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
        meanvarbuff[meanstashindex] = mean; // stash mean
    }
}

// This kernel is independent of others
// Partial summation of dBias = sum(dY)
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialDBias(const __global _FLOAT* __restrict dy_in,
                         __global _FLOAT* __restrict dbiasbuff)
{

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int xgid    = get_global_id(0);
    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    _FLOAT dbias      = 0.;

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            dbias += dy_in[index];
        }
    }

#ifdef __AMDGCN__

    lcl_data[ylid] = dbias;
    barrier(CLK_LOCAL_MEM_FENCE);
    _FLOAT tmp = 0.;
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    dbias = lcl_data[ylid];
    dbias += as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x111, 15, 15, 0));
    dbias += as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x112, 15, 15, 0));
    dbias += as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x114, 15, 15, 0));
    dbias += as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x142, 15, 15, 0));
    dbias += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x143, 15, 15, 0));
    dbias += tmp;

    if(ylid == 63)
    {
        unsigned int biasstashindex = cidx + ygrp_sz * ygrp_id + 6;
        dbiasbuff[biasstashindex]   = dbias;
    }

#else

    lcl_data[ylid] = dbias;
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
    dbias = lcl_data[0];

    if(ylid == 0)
    {
        unsigned int biasstashindex = cidx + ygrp_sz * ygrp_id + 6;
        dbiasbuff[biasstashindex]   = dbias;
    }
#endif

} // end spatial dbias kernel

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialFinalDBias(__global _FLOAT* buff, __global _FLOAT* delta_bias)
{

    _FLOAT db = 0.;

    unsigned int ylid    = get_local_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    int cidx             = MIO_BN_HW * xgid;

    for(int gn = 0; gn < yngrps; gn++)
    {
        unsigned int offset    = gn * ygrp_sz + ylid;
        unsigned int betaindex = cidx + ygrp_sz * offset + 6; // 1;
        if(offset < yngrps)
        { // modify to span larger number of groups
            db += buff[betaindex];
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_NGRPS > 64)

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    _FLOAT tmp     = 0.;
    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    db = lcl_data[ylid];
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;
    if(ygid == 63)
        delta_bias[xgid] = db;

#elif(MIO_BN_NGRPS > 16)

    _FLOAT tmp = 0.;
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;
    if(ygid == 63)
        delta_bias[xgid] = db;

#else
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x101, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x102, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x104, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x108, 15, 15, 0));
    if(ygid == 0)
        delta_bias[xgid] = db;
#endif

#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = db;
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
    db = lcl_data[0];

#else

    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        db += lcl_data[i];
    }

#endif

    if(ygid == 0)
        delta_bias[xgid] = db;

#endif
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialVariance(const __global _FLOAT* __restrict in,
                            __global _FLOAT* __restrict meanvarbuff)
{
    // SPATIAL
    _FLOAT mean     = 0.;
    _FLOAT elemStd  = 0.;
    _FLOAT variance = 0.;

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;

    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    mean                        = meanvarbuff[meanstashindex]; // load stashed mean

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
    variance = lcl_data[0];

    if(ylid == 0)
    {
        unsigned int varindex = cidx + ygrp_sz * ygrp_id + 2;
        meanvarbuff[varindex] = variance; // pre-stage for group reduction
    }

#endif

} // end spatial variance

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialFinalVariance(__global _FLOAT* __restrict varbuff, double epsilon)
{

    // SPATIAL
    __private _FLOAT variance    = 0.;
    __private _FLOAT invVariance = 0.;

    unsigned int ylid     = get_local_id(1);
    unsigned int ygrp_id  = get_group_id(1);
    unsigned int xgid     = get_global_id(0);
    unsigned int ygrp_sz  = get_local_size(1);
    unsigned int yngrps   = get_num_groups(1);
    unsigned int cidx     = xgid * MIO_BN_HW;
    unsigned int commitID = 0;

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

#pragma unroll
    for(int gn = 0; gn < MIO_BN_NGRPS; gn++)
    {
        unsigned int offset = gn * ygrp_sz + ylid;
        if(offset < yngrps)
        { // modify to span larger number of groups
            unsigned int varindex = cidx + ygrp_sz * offset + 2;
            variance += varbuff[varindex]; // load per group variance
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_NGRPS > 64)

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);

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
    commitID = 63;

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
    commitID   = 63;

#else
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x101, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x102, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x104, 15, 15, 0));
    variance += as_float(__builtin_amdgcn_mov_dpp(as_int(variance), 0x108, 15, 15, 0));
    commitID = 0;

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
    variance = lcl_data[0];

#else //(MIO_BN_NGRPS <= 16)

    variance = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        variance += lcl_data[i];
    }

#endif
    commitID = 0;

#endif

    variance /= NHW;
    invVariance = rsqrt(variance + epsilon);
    if(ylid == commitID)
    {
        unsigned int varstashindex = cidx + ygrp_sz * ygrp_id + 3;
        varbuff[varstashindex]     = invVariance; // stash
    }

} // end spatial final variance

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialDScale(const __global _FLOAT* x_in,
                          const __global _FLOAT* dy_in,
                          __global _FLOAT* buff)
{

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    int ylid    = get_local_id(1); // accumilate / reduction
    int ygrp_id = get_group_id(1);
    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int ygrp_sz = get_local_size(1);
    int cidx    = MIO_BN_HW * xgid;
    unsigned int index, ncIdx;

    _FLOAT mean    = 0.;
    _FLOAT invVar  = 0.;
    _FLOAT elemStd = 0.;
    _FLOAT xhat    = 0.;
    _FLOAT dscale  = 0.;

    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;

    mean   = buff[meanstashindex]; // load stashed mean
    invVar = buff[varstashindex];

    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            ncIdx   = n * MIO_BN_CHW + cidx;
            index   = ncIdx + ygid;
            elemStd = x_in[index] - mean; // (x_i - mean)
            xhat    = elemStd * invVar;
            dscale  = mad(xhat, dy_in[index], dscale);
        } // end for
    }     // end if

    lcl_data[ylid] = dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__

    _FLOAT tmp = 0.;
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    dscale = lcl_data[ylid];
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x111, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x112, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x114, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x142, 15, 15, 0));
    dscale += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x143, 15, 15, 0));
    dscale += tmp;
    if(ylid == 63)
    {
        unsigned int gammaindex = cidx + ygrp_sz * ygrp_id + 4;
        buff[gammaindex]        = dscale; // pre-stage for group reduction
    }

#else

    lcl_data[ylid] = dscale;
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
    dscale = lcl_data[0];
    if(ylid == 0)
    {
        unsigned int gammaindex = cidx + ygrp_sz * ygrp_id + 4;
        buff[gammaindex]        = dscale;
    }

#endif
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialFinalDScale(__global _FLOAT* buff, __global _FLOAT* delta_scale)
{

    __private _FLOAT dscale = 0.;

    unsigned int ylid    = get_local_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    int cidx             = MIO_BN_HW * xgid;

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

#pragma unroll
    for(int gn = 0; gn < MIO_BN_NGRPS; gn++)
    {
        unsigned int offset = gn * ygrp_sz + ylid;
        if(offset < yngrps)
        { // modify to span larger number of groups
            unsigned int gammaindex = cidx + ygrp_sz * offset + 4;
            dscale += buff[gammaindex];
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_NGRPS > 64)

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    _FLOAT tmp = 0.;

    lcl_data[ylid] = dscale;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    dscale = lcl_data[ylid];
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x111, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x112, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x114, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x142, 15, 15, 0));
    dscale += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x143, 15, 15, 0));
    dscale += tmp;
    if(ygid == 63)
        delta_scale[xgid] = dscale / NHW;

#elif(MIO_BN_NGRPS > 16)
    _FLOAT tmp = 0.;
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x111, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x112, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x114, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x142, 15, 15, 0));
    dscale += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x143, 15, 15, 0));
    dscale += tmp;
    if(ygid == 63)
        delta_scale[xgid] = dscale / NHW;

#else
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x101, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x102, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x104, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x108, 15, 15, 0));
    if(ygid == 0)
        delta_scale[xgid] = dscale / NHW;

#endif

#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = dscale;
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
    dscale = lcl_data[0];

#else //(MIO_BN_NGRPS <= 16)

    dscale = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        dscale += lcl_data[i];
    }

#endif
    if(ygid == 0)
        delta_scale[xgid] = dscale / NHW;

#endif
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialDX(const __global _FLOAT* x_in,
                      const __global _FLOAT* dy_in,
                      __global _FLOAT* dx_out,
                      const __global _FLOAT* bnScale,
                      __global _FLOAT* delta_scale,
                      __global _FLOAT* delta_bias)
{

    int ygrp_id = get_group_id(1);
    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int ygrp_sz = get_local_size(1);
    int cidx    = MIO_BN_HW * xgid;
    unsigned int index;
    _FLOAT mean, invVar;
    _FLOAT elemStd, xhat;
    _FLOAT scale, dscale, dbias;
    _FLOAT tmp1, tmp2, tmp3;
    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    local _FLOAT lscale, ldscale, ldbias, lmean, livar;
    unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
    unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;

    if(get_local_id(1) == 0)
    {
        lmean   = dx_out[meanstashindex]; // load stashed mean
        livar   = dx_out[varstashindex];
        lscale  = bnScale[xgid];
        ldscale = delta_scale[xgid];
        ldbias  = delta_bias[xgid];
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

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index         = n * MIO_BN_CHW + cidx + ygid;
            elemStd       = x_in[index] - mean; // (x_i - mean)
            xhat          = elemStd * invVar;   // recalculating this again...
            tmp1          = mad(NHW, dy_in[index], -dbias);
            tmp2          = -xhat * dscale;
            tmp3          = (scale * invVar) / NHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1); // DEBUG
        }
    }
}

//============================================================

#else

//===================== SPATIAL SAVED ========================

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSavedDScale(const __global _FLOAT* x_in,
                               const __global _FLOAT* dy_in,
                               const __global _FLOAT* savedMean,
                               const __global _FLOAT* savedInvVariance,
                               __global _FLOAT* dscalebuff)
{

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    int ylid    = get_local_id(1); // accumilate / reduction
    int ygrp_id = get_group_id(1);
    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int ygrp_sz = get_local_size(1);
    int cidx    = MIO_BN_HW * xgid;
    unsigned int index;

    _FLOAT mean, invVar;
    _FLOAT elemStd, xhat;
    _FLOAT dscale = 0.;

    mean   = savedMean[xgid];
    invVar = savedInvVariance[xgid];

    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index   = n * MIO_BN_CHW + cidx + ygid;
            elemStd = x_in[index] - mean; // (x_i - mean)
            xhat    = elemStd * invVar;
            dscale  = mad(xhat, dy_in[index], dscale);
        } // end for
    }     // end if
    lcl_data[ylid] = dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__

    _FLOAT tmp = 0.;
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    dscale = lcl_data[ylid];
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x111, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x112, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x114, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x142, 15, 15, 0));
    dscale += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x143, 15, 15, 0));
    dscale += tmp;

    if(ylid == 63)
    {
        unsigned int gammaindex = cidx + ygrp_sz * ygrp_id;
        dscalebuff[gammaindex]  = dscale; // pre-stage for group reduction
    }

#else

    lcl_data[ylid] = dscale;
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
    dscale = lcl_data[0];
    if(ylid == 0)
    {
        unsigned int gammaindex = cidx + ygrp_sz * ygrp_id;
        dscalebuff[gammaindex]  = dscale;
    }

#endif
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSavedFinalDScale(__global _FLOAT* buff, __global _FLOAT* delta_scale)
{

    __private _FLOAT dscale = 0.;

    unsigned int ylid    = get_local_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    int cidx             = MIO_BN_HW * xgid;

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

#pragma unroll
    for(int gn = 0; gn < MIO_BN_NGRPS; gn++)
    {
        unsigned int offset = gn * ygrp_sz + ylid;
        if(offset < yngrps)
        { // modify to span larger number of groups
            unsigned int gammaindex = cidx + ygrp_sz * offset;
            dscale += buff[gammaindex];
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_NGRPS > 64)

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    _FLOAT tmp = 0.;

    lcl_data[ylid] = dscale;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    dscale = lcl_data[ylid];
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x111, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x112, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x114, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x142, 15, 15, 0));
    dscale += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x143, 15, 15, 0));
    dscale += tmp;
    if(ygid == 63)
        delta_scale[xgid] = dscale / NHW;

#elif(MIO_BN_NGRPS > 16)
    _FLOAT tmp = 0.;
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x111, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x112, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x114, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x142, 15, 15, 0));
    dscale += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x143, 15, 15, 0));
    dscale += tmp;
    if(ygid == 63)
        delta_scale[xgid] = dscale / NHW;

#else
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x101, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x102, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x104, 15, 15, 0));
    dscale += as_float(__builtin_amdgcn_mov_dpp(as_int(dscale), 0x108, 15, 15, 0));
    if(ygid == 0)
        delta_scale[xgid] = dscale / NHW;

#endif

#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = dscale;
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
    dscale = lcl_data[0];
#else //(MIO_BN_NGRPS <= 16)

    dscale = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        dscale += lcl_data[i];
    }
#endif
    if(ygid == 0)
        delta_scale[xgid] = dscale / NHW;

#endif
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSavedDBias(const __global _FLOAT* __restrict dy_in,
                              __global _FLOAT* __restrict dbiasbuff)
{

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    unsigned int ylid    = get_local_id(1);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    _FLOAT dbias      = 0.;

    // move across the sections of the image mini_batch stack

    if(ygid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index = n * MIO_BN_CHW + cidx + ygid;
            dbias += dy_in[index];
        }
    }
    lcl_data[ylid] = dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__

    _FLOAT tmp = 0.;
    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    dbias = lcl_data[ylid];
    dbias += as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x111, 15, 15, 0));
    dbias += as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x112, 15, 15, 0));
    dbias += as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x114, 15, 15, 0));
    dbias += as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x142, 15, 15, 0));
    dbias += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(dbias), 0x143, 15, 15, 0));
    dbias += tmp;

    if(ylid == 63)
    {
        unsigned int biasstashindex = cidx + ygrp_sz * ygrp_id + 1;
        dbiasbuff[biasstashindex]   = dbias;
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
    dbias = lcl_data[0];
    if(ylid == 0)
    {
        unsigned int biasstashindex = cidx + ygrp_sz * ygrp_id + 1;
        dbiasbuff[biasstashindex]   = dbias;
    }

#endif
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSavedFinalDBias(__global _FLOAT* buff, __global _FLOAT* delta_bias)
{

    _FLOAT db = 0.;

    unsigned int ylid    = get_local_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    int cidx             = MIO_BN_HW * xgid;

    for(int gn = 0; gn < yngrps; gn++)
    {
        unsigned int offset    = gn * ygrp_sz + ylid;
        unsigned int betaindex = cidx + ygrp_sz * offset + 1;
        if(offset < yngrps)
        { // modify to span larger number of groups
            db += buff[betaindex];
        }
    }

#ifdef __AMDGCN__

#if(MIO_BN_NGRPS > 64)

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    _FLOAT tmp     = 0.;
    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ylid < 128)
        lcl_data[ylid] += lcl_data[ylid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < 64)
        lcl_data[ylid] += lcl_data[ylid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);
    db = lcl_data[ylid];
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;
    if(ygid == 63)
        delta_bias[xgid] = db;

#elif(MIO_BN_NGRPS > 16)

    _FLOAT tmp = 0.;
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x111, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x112, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x114, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x118, 15, 15, 0));
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x142, 15, 15, 0));
    db += tmp;
    tmp = as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x143, 15, 15, 0));
    db += tmp;
    if(ygid == 63)
        delta_bias[xgid] = db;

#else
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x101, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x102, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x104, 15, 15, 0));
    db += as_float(__builtin_amdgcn_mov_dpp(as_int(db), 0x108, 15, 15, 0));
    if(ygid == 0)
        delta_bias[xgid] = db;
#endif

#else

    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = db;
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
    db = lcl_data[0];
#else //(MIO_BN_NGRPS <= 16)

    db = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        db += lcl_data[i];
    }
#endif
    if(ygid == 0)
        delta_bias[xgid] = db;

#endif
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialSavedDX(const __global _FLOAT* x_in,
                           const __global _FLOAT* dy_in,
                           __global _FLOAT* dx_out,
                           const __global _FLOAT* bnScale,
                           __global _FLOAT* delta_scale,
                           __global _FLOAT* delta_bias,
                           const __global _FLOAT* savedMean,
                           const __global _FLOAT* savedInvVariance)
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
        lmean   = savedMean[xgid];        // load stashed mean
        livar   = savedInvVariance[xgid]; // load stashed inverse variance
        lscale  = bnScale[xgid];
        ldscale = delta_scale[xgid];
        ldbias  = delta_bias[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            index         = n * MIO_BN_CHW + cidx + ygid;
            elemStd       = x_in[index] - mean; // (x_i - mean)
            xhat          = elemStd * invVar;   // recalculating this again...
            tmp1          = mad(NHW, dy_in[index], -dbias);
            tmp2          = -xhat * dscale;
            tmp3          = (scale * invVar) / NHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1); // DEBUG
        }
    }
}

//============================================================

#endif

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
