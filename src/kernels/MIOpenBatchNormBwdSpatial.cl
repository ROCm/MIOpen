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

#ifndef MIO_BN_NLOOP
#define MIO_BN_NLOOP 1
#endif

#ifndef MIO_BN_USESAVED
#define MIO_BN_USESAVED 1
#endif

#define MIO_BN_MAXN 512

#ifndef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#elif(MIO_BN_NODPP == 1)
#undef __AMDGCN__
#endif

/*
#ifdef __AMDGCN__
#undef __AMDGCN__
#endif
*/

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

#define UNUSED __attribute__((__unused__))

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

#if(MIO_BN_VARIANT == 0)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatial(const __global _FLOAT* __restrict x_in,
                    const __global _FLOAT* __restrict dy_in,
                    __global _FLOAT* __restrict dx_out,
                    const __global _FLOAT* bnScale,
                    __global _FLOAT* __restrict dscale,
                    __global _FLOAT* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                    double epsilon,
#elif(MIO_BN_USESAVED == 1)
                    const __global _FLOAT* savedMean,
                    const __global _FLOAT* savedInvVariance,
#endif
                    float INHW)
{

    // SPATIAL
    _FLOAT mean = 0.;
#if(MIO_BN_USESAVED == 0)
    _FLOAT variance = 0.;
#endif
    _FLOAT invVar = 0.;
    _FLOAT pscale = 0.;
    _FLOAT ds     = 0.;
    _FLOAT db     = 0.;

    _FLOAT batchvalues[MIO_BN_N];
    _FLOAT dyvalues[MIO_BN_N];

    __local _FLOAT lbns;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
#if(MIO_BN_USESAVED == 1)
    __local _FLOAT lmean, lvar;
#endif
    unsigned int index  = 0;
    unsigned int lid    = get_local_id(0);
    unsigned int grpid  = get_group_id(0);
    unsigned int chwid  = grpid * MIO_BN_HW + (lid % MIO_BN_HW);
    unsigned int lidihw = lid / MIO_BN_HW;
    unsigned int segihw = MIO_BN_SEGMENT / MIO_BN_HW;
    unsigned int nid    = 0;
    _FLOAT tmp1, tmp2, tmp3;

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    if(lid == 0)
    {
        lbns = bnScale[grpid];

#if(MIO_BN_USESAVED == 1)

        lmean = savedMean[grpid];
        lvar  = savedInvVariance[grpid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean   = lmean;
    invVar = lvar;
#else // recalc mean and variance

    } // end if(!lid)

    if(lid < MIO_BN_SEGMENT)
    {
//==== CALC MEAN =======================
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_NLOOP; n++)
        {
            nid            = n * segihw + lidihw;
            index          = nid * MIO_BN_CHW + chwid;
            batchvalues[n] = (index < MIO_BN_NCHW) ? x_in[index] : 0.;
            mean += batchvalues[n];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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
            nid = n * segihw + lidihw;
            if(nid < MIO_BN_N)
            {
                batchvalues[n] = (batchvalues[n] - mean);
                variance       = mad(batchvalues[n], batchvalues[n], variance);
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

    invVar = rsqrt(variance + epsilon);
// DONE VARIANCE

#endif //! USESAVED

    if(lid < MIO_BN_SEGMENT)
    {
//==== CALC DB and DS =========================================
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_NLOOP; n++)
        {
            nid   = n * segihw + lidihw;
            index = nid * MIO_BN_CHW + chwid;
            if(index < MIO_BN_NCHW)
            {
                dyvalues[n] = dy_in[index];
                db += dyvalues[n];
#if(MIO_BN_USESAVED == 1)
                batchvalues[n] = (x_in[index] - mean);
#endif
                batchvalues[n] = (batchvalues[n] * invVar);
                ds             = mad(batchvalues[n], dyvalues[n], ds);
            }
            else
            {
#if(MIO_BN_USESAVED == 0)
                batchvalues[n] = 0;
#endif
                dyvalues[n] = 0.;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&db, lcl_data, lid, 1);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&db, lcl_data, lid, 1);
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&ds, lcl_data, lid, 1);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&ds, lcl_data, lid, 1);
#endif

    if(lid < MIO_BN_SEGMENT)
    {
        // Group level reduction
        // Need to reduce over all elements in NxHxW
        // move across the sections of an image in the mini_batch stack
        pscale = lbns;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_NLOOP; n++)
        {
            nid   = n * segihw + lidihw;
            index = nid * MIO_BN_CHW + chwid;
            if(index < MIO_BN_NCHW)
            {
                tmp1          = mad(NHW, dyvalues[n], -db);
                tmp2          = -batchvalues[n] * ds;
                tmp3          = (pscale * invVar) * INHW;
                dx_out[index] = tmp3 * (tmp2 + tmp1);
            }
        }
    }
    if(lid == 0)
    {
        dbias[grpid]  = db;
        dscale[grpid] = ds;
    }

} // end spatial

#elif(MIO_BN_VARIANT == 1)

#ifdef __AMDGCN__
#undef __AMDGCN__
#endif

//=============== SINGLE WORKGROUP PER CHANNEL

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatial(const __global _FLOAT* __restrict x_in,
                    const __global _FLOAT* __restrict dy_in,
                    __global _FLOAT* __restrict dx_out,
                    const __global _FLOAT* bnScale,
                    __global _FLOAT* __restrict dscale,
                    __global _FLOAT* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                    double epsilon,
#elif(MIO_BN_USESAVED == 1)
                    const __global _FLOAT* savedMean,
                    const __global _FLOAT* savedInvVariance,
#endif
                    float INHW)
{

    // SPATIAL
    _FLOAT mean     = 0.;
#if(MIO_BN_USESAVED == 0)
    _FLOAT variance = 0.;
#endif
    _FLOAT invVar   = 0.;
    _FLOAT pscale   = 0.;
    _FLOAT elemStd  = 0.;
    _FLOAT ds       = 0.;
    _FLOAT db       = 0.;
    _FLOAT batchvalues[MIO_BN_N];
    _FLOAT dyvalues[MIO_BN_N];
    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    _FLOAT tmp1, tmp2, tmp3;

#if(MIO_BN_USESAVED == 1)
    __local _FLOAT lmean, lvar;
#endif

#ifndef __AMDGCN__
#if(MIO_BN_HW > 1)
    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
#endif
#endif

    __local _FLOAT lcl_scale;

    if(ylid == 0)
    {
        lcl_scale = bnScale[xgid];

#if(MIO_BN_USESAVED == 1)

        lmean = savedMean[xgid];
        lvar  = savedInvVariance[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean   = lmean;
    invVar = lvar;
#else // recalc mean and variance

    } // end if(!lid)

    if(ylid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index          = n * MIO_BN_CHW + cidx + ylid;
            batchvalues[n] = x_in[index];
            mean += batchvalues[n];
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
            batchvalues[n] = batchvalues[n] - mean;
            variance       = mad(batchvalues[n], batchvalues[n], variance);
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
    invVar = rsqrt(variance + epsilon);
#endif // !useSaved
    if(ylid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index       = n * MIO_BN_CHW + cidx + ylid;
            dyvalues[n] = dy_in[index];
            db += dyvalues[n];
#if(MIO_BN_USESAVED == 1)
            batchvalues[n] = (x_in[index] - mean);
#endif
            batchvalues[n] = (batchvalues[n] * invVar);
            ds             = mad(batchvalues[n], dyvalues[n], ds);
        }
    }

//===dBias reduction=======================
#ifdef __AMDGCN__

#if(MIO_BN_HW > 16)
    dppRegReduce64(&db, 1);
#elif(MIO_BN_HW > 1)
    dppRegReduce16(&db, 1);
#endif // HW
#else  // GCN

    barrier(CLK_LOCAL_MEM_FENCE);
#if(MIO_BN_HW > 16)
    regLDSreduce(&db, lcl_data, ylid, 1);
#elif(MIO_BN_HW > 1)
    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = 0.;
    for(int i = 0; i < MIO_BN_HW; i++)
    {
        db += lcl_data[i];
    }
#endif // HW
#endif // GCN
//==========================================

//===dScale reduction=======================
#ifdef __AMDGCN__

#if(MIO_BN_HW > 16)
    dppRegReduce64(&ds, 1);
#elif(MIO_BN_HW > 1)
    dppRegReduce16(&ds, 1);
#endif // HW
#else  // if not GCN

#if(MIO_BN_HW > 16)
    regLDSreduce(&ds, lcl_data, ylid, 1);
#elif(MIO_BN_HW > 1)
    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_HW; i++)
    {
        ds += lcl_data[i];
    }
#endif // HW
#endif // GCN
    //===========================================

    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ylid < MIO_BN_HW)
    {
        pscale = lcl_scale;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index         = n * MIO_BN_CHW + cidx + ylid;
            tmp1          = mad(NHW, dyvalues[n], -db);
            tmp2          = -(batchvalues[n]) * ds;
            tmp3          = (pscale * invVar) * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1);
        }
    }
    if(ylid == 0)
    {
        dbias[xgid]  = db;
        dscale[xgid] = ds;
    }

} // end spatial

#elif(MIO_BN_VARIANT == 2)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatial(const __global _FLOAT* __restrict x_in,
                    const __global _FLOAT* __restrict dy_in,
                    __global _FLOAT* __restrict dx_out,
                    const __global _FLOAT* bnScale,
                    __global _FLOAT* __restrict dscale,
                    __global _FLOAT* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                    double epsilon,
#elif(MIO_BN_USESAVED == 1)
                    const __global _FLOAT* savedMean,
                    const __global _FLOAT* savedInvVariance,
#endif
                    float INHW)
{

    // SPATIAL
    _FLOAT mean     = 0.;
#if(MIO_BN_USESAVED == 0)
    _FLOAT variance = 0.;
#endif
    _FLOAT invVar   = 0.;
    _FLOAT pscale   = 0.;
    _FLOAT elemStd  = 0.;
    _FLOAT ds       = elemStd;
    _FLOAT db       = 0.;
    _FLOAT batchvalues[MIO_BN_HW];
    _FLOAT dyvalues[MIO_BN_HW];
    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    _FLOAT tmp1, tmp2, tmp3;

#if(MIO_BN_USESAVED == 1)
    __local _FLOAT lmean, lvar;
#endif

#if(MIO_BN_N > 1 && MIO_BN_N < 17)
    local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
#endif

    __local _FLOAT lcl_scale;

    if(ylid == 0)
    {
        lcl_scale = bnScale[xgid];

#if(MIO_BN_USESAVED == 1)

        lmean = savedMean[xgid];
        lvar  = savedInvVariance[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean   = lmean;
    invVar = lvar;
#else // recalc mean and variance

    } // end if(!lid)
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < MIO_BN_N)
    {
#pragma unroll
        for(int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index           = ylid * MIO_BN_CHW + cidx + hw;
            batchvalues[hw] = x_in[index];
            mean += batchvalues[hw];
        }
    }
    else
    {
        mean = 0.;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef __AMDGCN__

#if(MIO_BN_N > 16)
    dppRegReduce64(&mean, INHW);
#elif(MIO_BN_N > 1)
    lcl_data[ylid] = mean;
    barrier(CLK_LOCAL_MEM_FENCE);
    mean = 0.;
    for(int i = 0; i < MIO_BN_N; i++)
    {
        mean += lcl_data[i];
    }
    mean *= INHW;
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
    for(int i = 0; i < MIO_BN_N; i++)
    {
        mean += lcl_data[i];
    }
    mean *= INHW;
#else
    mean *= INHW;
#endif // N
#endif // GCN
    barrier(CLK_LOCAL_MEM_FENCE);
    if(ylid < MIO_BN_N)
    {
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            batchvalues[hw] = batchvalues[hw] - mean;
            variance        = mad(batchvalues[hw], batchvalues[hw], variance);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef __AMDGCN__

#if(MIO_BN_N > 16)
    dppRegReduce64(&variance, INHW);
#elif(MIO_BN_N > 1)
    lcl_data[ylid] = variance;
    barrier(CLK_LOCAL_MEM_FENCE);
    variance = 0.;
    for(int i = 0; i < MIO_BN_N; i++)
    {
        variance += lcl_data[i];
    }
    variance *= INHW;

#else
    variance *= INHW;
#endif // N

#else // if not GCN

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
    invVar = rsqrt(variance + epsilon);
#endif // !useSaved

    if(ylid < MIO_BN_N)
    {
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index        = ylid * MIO_BN_CHW + cidx + hw;
            dyvalues[hw] = dy_in[index];
            db += dyvalues[hw];

#if(MIO_BN_USESAVED == 1)
            batchvalues[hw] = (x_in[index] - mean);
#endif
            batchvalues[hw] = (batchvalues[hw] * invVar);
            ds              = mad(batchvalues[hw], dyvalues[hw], ds);
        }
    }
    else
    {
        ds = 0.;
        db = 0.;
    }
//===dBias reduction=======================
#ifdef __AMDGCN__
#if(MIO_BN_N > 16)
    dppRegReduce64(&db, 1);
#elif(MIO_BN_N > 1)
    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = 0.;
    for(int i = 0; i < MIO_BN_N; i++)
    {
        db += lcl_data[i];
    }
#endif // N

#else // GCN

#if(MIO_BN_N > 16)
    regLDSreduce(&db, lcl_data, ylid, 1);
#elif(MIO_BN_N > 1)
    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = 0.;
    for(int i = 0; i < MIO_BN_N; i++)
    {
        db += lcl_data[i];
    }
#endif // N
#endif // GCN
//==========================================
//===dScale reduction=======================
#ifdef __AMDGCN__
#if(MIO_BN_N > 16)
    dppRegReduce64(&ds, 1);
#elif(MIO_BN_N > 1)
    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_N; i++)
    {
        ds += lcl_data[i];
    }
#endif // N
#else  // if not GCN

#if(MIO_BN_N > 16)
    regLDSreduce(&ds, lcl_data, ylid, 1);
#elif(MIO_BN_N > 1)
    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_N; i++)
    {
        ds += lcl_data[i];
    }
#endif // HW
#endif // GCN
    //===========================================
    barrier(CLK_LOCAL_MEM_FENCE);
    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ylid < MIO_BN_N)
    {
        pscale = lcl_scale;
#pragma unroll
        for(unsigned int hw = 0; hw < MIO_BN_HW; hw++)
        {
            index         = ylid * MIO_BN_CHW + cidx + hw;
            tmp1          = mad(NHW, dyvalues[hw], -db);
            tmp2          = -(batchvalues[hw]) * ds;
            tmp3          = (pscale * invVar) * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1);
        }
    }
    if(ylid == 0)
    {
        dbias[xgid]  = db;
        dscale[xgid] = ds;
    }

} // end spatial

#elif(MIO_BN_VARIANT == 3)

//=============== SINGLE WORKGROUP PER CHANNEL

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatial(const __global _FLOAT* __restrict x_in,
                    const __global _FLOAT* __restrict dy_in,
                    __global _FLOAT* __restrict dx_out,
                    const __global _FLOAT* bnScale,
                    __global _FLOAT* __restrict dscale,
                    __global _FLOAT* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                    double epsilon,
#elif(MIO_BN_USESAVED == 1)
                    const __global _FLOAT* savedMean,
                    const __global _FLOAT* savedInvVariance,
#endif
                    float INHW)
{

    // SPATIAL
    _FLOAT mean     = 0.;
#if(MIO_BN_USESAVED == 0)
    _FLOAT variance = 0.;
#endif
    _FLOAT invVar   = 0.;
    _FLOAT pscale   = 0.;
    _FLOAT elemStd  = 0.;
    _FLOAT ds       = elemStd;
    _FLOAT db       = 0.;
#if(MIO_BN_N < MIO_BN_MAXN)
    _FLOAT batchvalues[MIO_BN_N];
    _FLOAT dyvalues[MIO_BN_N];
#endif

#ifdef __AMDGCN__
    unsigned int segment = MIO_BN_GRP1 >> 6;
#endif

    unsigned int ylid = get_local_id(1);
    unsigned int xgid = get_global_id(0);
    unsigned int index;
    unsigned int cidx = xgid * MIO_BN_HW;
    _FLOAT tmp1, tmp2, tmp3;

#if(MIO_BN_USESAVED == 1)
    __local _FLOAT lmean, lvar;
#endif

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    __local _FLOAT lcl_scale;

    if(ylid == 0)
    {
        lcl_scale = bnScale[xgid];

#if(MIO_BN_USESAVED == 1)

        lmean = savedMean[xgid];
        lvar  = savedInvVariance[xgid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean   = lmean;
    invVar = lvar;
#else // recalc mean and variance

    } // end if(!lid)

    if(ylid < MIO_BN_HW)
    {
#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index                  = n * MIO_BN_CHW + cidx + ylid;
#if(MIO_BN_N < MIO_BN_MAXN)
            mean += batchvalues[n] = x_in[index];
#else
            mean += x_in[index];
#endif
        }
    }
    else
    {
        mean = 0.;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
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
            batchvalues[n] = batchvalues[n] - mean; //(in[index] - mean);
            variance       = mad(batchvalues[n], batchvalues[n], variance);
#else
            index    = n * MIO_BN_CHW + xgid * MIO_BN_HW + ylid;
            elemStd  = (in[index] - mean);
            variance = mad(elemStd, elemStd, variance);
#endif
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

    invVar = rsqrt(variance + epsilon);

#endif // !useSaved

    if(ylid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index             = n * MIO_BN_CHW + cidx + ylid;
#if(MIO_BN_N < MIO_BN_MAXN)
            db += dyvalues[n] = dy_in[index];

#if(MIO_BN_USESAVED == 1)
            batchvalues[n]    = (x_in[index] - mean);
#endif
            batchvalues[n]    = (batchvalues[n] * invVar);
            ds                = mad(batchvalues[n], dyvalues[n], ds);
#else // maxn
            db += dy_in[index];
            _FLOAT elemStd
#if(MIO_BN_USESAVED == 1)
                elemStd = (x_in[index] - mean);
#endif
            elemStd      = (elemStd * invVar);
            ds           = mad(elemStd, dy_in[index], ds);
#endif
        }
    }
    else
    {
        db = 0.;
        ds = 0.;
    }

    //===dBias reduction=======================
    // barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[ylid] = db;
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
    db = 0.;
    if(ylid < 64)
    {
        for(unsigned int red = 0; red < segment; red++)
        {
            db += lcl_data[ylid * segment + red];
        }
    }
    else
    {
        db = 0.;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    dppLDSReduce64(&db, lcl_data, ylid, 1);
#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&db, lcl_data, ylid, 1);
#endif
    //==========================================

    //===dScale reduction=======================
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
    ds = 0.;
    if(ylid < 64)
    {
        for(unsigned int red = 0; red < segment; red++)
        {
            ds += lcl_data[ylid * segment + red];
        }
    }
    else
    {
        ds = 0.;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    dppLDSReduce64(&ds, lcl_data, ylid, 1);

#else
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&ds, lcl_data, ylid, 1);
#endif
    //===========================================

    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ylid < MIO_BN_HW)
    {
        pscale = lcl_scale;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index         = n * MIO_BN_CHW + cidx + ylid;
#if(MIO_BN_N < MIO_BN_MAXN)
            tmp1          = mad(NHW, dyvalues[n], -db);
            tmp2          = -(batchvalues[n]) * ds;
#else
            tmp1 = mad(NHW, dy_in[index], -db);
            tmp2 = -(x_in[index] - mean) * invVar * ds;
#endif
            tmp3          = (pscale * invVar) * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1);
        }
    }
    if(ylid == 0)
    {
        dbias[xgid]  = db;
        dscale[xgid] = ds;
    }

} // end spatial

#elif(MIO_BN_VARIANT == 4)

#if(MIO_BN_USESAVED == 0)
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

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialFinalMean(__global _FLOAT* __restrict meanvarbuff, float INHW)
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
    dppRegReduce64(&mean, INHW);
    commitID = 63;
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
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialVariance(const __global _FLOAT* __restrict in, /* x input */
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
BatchNormBwdSpatialFinalVariance(__global _FLOAT* __restrict varbuff, float INHW, double epsilon)
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
}

#endif // end USESAVED == 0

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

    lcl_data[ylid] = dbias;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&dbias, lcl_data, ylid, 1);

#else
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&dbias, lcl_data, ylid, 1);

#endif

    if(ylid == 0)
    {
        unsigned int biasstashindex = cidx + ygrp_sz * ygrp_id + 6;
        dbiasbuff[biasstashindex]   = dbias;
    }
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

#if(MIO_BN_NGRPS > 64)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&db, lcl_data, ylid, 1);
#else  // GCN
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&db, lcl_data, ylid, 1);
#endif // GCN

#elif(MIO_BN_NGRPS <= 64)
#ifdef __AMDGCN__
    dppRegReduce64(&db, 1);
#else // GCN
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    regLDSreduce(&db, lcl_data, ylid, 1);
#endif // GCN
#else  // NGRPS
#ifdef __AMDGCN__
    dppRegReduce16(&db, 1);
#else // GCN
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
    db = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        db += lcl_data[i];
    }
#endif // GCN
#endif // NGRPS

    if(ygid == 0)
        delta_bias[xgid] = db;
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialDScale(const __global _FLOAT* x_in,
                          const __global _FLOAT* dy_in,
                          __global _FLOAT* buff
#if(MIO_BN_USESAVED == 1)

                          ,
                          const __global _FLOAT* savedMean,
                          const __global _FLOAT* savedInvVariance
#endif
                          )
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

    __local _FLOAT lmean, livar;

    if(ylid == 0)
    {
#if(MIO_BN_USESAVED == 0)
        unsigned int meanstashindex = cidx + ygrp_sz * ygrp_id + 1;
        unsigned int varstashindex  = cidx + ygrp_sz * ygrp_id + 3;
        lmean                       = buff[meanstashindex]; // load stashed mean
        livar                       = buff[varstashindex];
#else  // NO SAVED
        lmean = savedMean[xgid];
        livar = savedInvVariance[xgid];
#endif // SAVED
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(ygid < MIO_BN_HW)
    {
        mean   = lmean;
        invVar = livar;
#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        { // apply normalization
            ncIdx   = n * MIO_BN_CHW + cidx;
            index   = ncIdx + ygid;
            elemStd = x_in[index] - mean; // (x_i - mean)
            xhat    = elemStd * invVar;
            dscale  = mad(xhat, dy_in[index], dscale);
            // dscale += 1.;
        } // end for
    }     // end if

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[ylid] = dscale;
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&dscale, lcl_data, ylid, 1);

#else // GCN
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&dscale, lcl_data, ylid, 1);

#endif // GCN
    if(ylid == 0)
    {
        unsigned int gammaindex = cidx + ygrp_sz * ygrp_id + 4;
        buff[gammaindex]        = dscale;
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialFinalDScale(__global _FLOAT* buff, __global _FLOAT* delta_scale)
{

    __private _FLOAT ds  = 0.;
    unsigned int ylid    = get_local_id(1);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int yngrps  = get_num_groups(1);
    int cidx             = MIO_BN_HW * xgid;

#pragma unroll
    for(int gn = 0; gn < MIO_BN_NGRPS; gn++)
    {
        unsigned int offset = gn * ygrp_sz + ylid;
        if(offset < yngrps)
        { // modify to span larger number of groups
            unsigned int gammaindex = cidx + ygrp_sz * offset + 4;
            ds += buff[gammaindex];
        }
    }

#if(MIO_BN_NGRPS > 64)
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 32; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&ds, lcl_data, ylid, 1);
#else  // GCN
    for(unsigned int red = (MIO_BN_GRP1 >> 1); red > 256; red >>= 1)
    {
        if(ylid < red)
            lcl_data[ylid] += lcl_data[ylid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&ds, lcl_data, ylid, 1);
#endif // GCN

#elif(MIO_BN_NGRPS <= 64)

#ifdef __AMDGCN__
    dppRegReduce64(&ds, 1);
#else // GCN
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    regLDSreduce(&ds, lcl_data, ylid, 1);
#endif // GCN
#else  // else < 16

#ifdef __AMDGCN__
    dppRegReduce16(&ds, 1);
#else // GCN
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];
    lcl_data[ylid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
    ds = 0.;
#pragma unroll
    for(int i = 0; i < MIO_BN_NGRPS; i++)
    {
        ds += lcl_data[i];
    }
#endif // end AMDGCN
#endif // NGRPS

    if(ygid == 0)
        delta_scale[xgid] = ds;
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatialDX(const __global _FLOAT* x_in,
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
        lmean                       = dx_out[meanstashindex]; // load stashed mean
        livar                       = dx_out[varstashindex];
#else  // SAVED
        lmean = savedMean[xgid];
        livar = savedInvVariance[xgid];
#endif // SAVED
        lscale                      = bnScale[xgid];
        ldscale                     = delta_scale[xgid];
        ldbias                      = delta_bias[xgid];
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
            tmp3          = scale * invVar * INHW;
            dx_out[index] = tmp3 * (tmp2 + tmp1);
        }
    }
}

//============================================================

#elif(MIO_BN_VARIANT == 5)

#define MIO_BN_REM (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_GRP0) * MIO_BN_GRP0))
#define MIO_BN_LESS (MIO_BN_NHW - MIO_BN_REM)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatial(const __global _FLOAT* __restrict x_in,
                    const __global _FLOAT* __restrict dy_in,
                    __global _FLOAT* __restrict dx_out,
                    const __global _FLOAT* bnScale,
                    __global _FLOAT* __restrict dscale,
                    __global _FLOAT* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                    double epsilon,
#elif(MIO_BN_USESAVED == 1)
                    const __global _FLOAT* savedMean,
                    const __global _FLOAT* savedInvVariance,
#endif
                    float INHW)
{

    // SPATIAL
    _FLOAT mean   = 0.;
    _FLOAT invVar = 0.;
    _FLOAT pscale = 0.;
    _FLOAT db     = 0.;
    _FLOAT ds     = 0.;

#if(MIO_BN_USESAVED == 1)
    __local _FLOAT lmean, lvar;
#endif

    __local _FLOAT lcl_scale;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

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
    _FLOAT variance = 0.;
    //==== CALC MEAN =======================
    mean = 0.;

#pragma unroll
    for(unsigned int k = lid; k < MIO_BN_LESS; k += MIO_BN_GRP0)
    {
        nidx  = iDiv(k, MIO_BN_HW);
        hwidx = iMod(k, nidx, MIO_BN_HW);
        index = nidx * MIO_BN_CHW + chwid + hwidx;
        mean += *(x_in + index);
    }
#if(MIO_BN_REM)
    nidx  = iDiv(MIO_BN_LESS + lid, MIO_BN_HW);
    hwidx = iMod(MIO_BN_LESS + lid, nidx, MIO_BN_HW);
    index = nidx * MIO_BN_CHW + grpid * MIO_BN_HW + hwidx;
    mean += (index < MIO_BN_NCHW) ? *(x_in + index) : 0.;
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
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
    _FLOAT diff = 0.;
    variance    = 0.;
#pragma unroll
    for(unsigned int k = lid; k < MIO_BN_LESS; k += MIO_BN_GRP0)
    {
        nidx     = iDiv(k, MIO_BN_HW);
        hwidx    = iMod(k, nidx, MIO_BN_HW);
        index    = nidx * MIO_BN_CHW + chwid + hwidx;
        diff     = *(x_in + index) - mean;
        variance = mad(diff, diff, variance);
    }

#if(MIO_BN_REM)
    nidx  = iDiv(MIO_BN_LESS + lid, MIO_BN_HW);
    hwidx = iMod(MIO_BN_LESS + lid, nidx, MIO_BN_HW);
    index = nidx * MIO_BN_CHW + chwid + hwidx;
    if(index < MIO_BN_NCHW)
    {
        diff     = *(x_in + index) - mean;
        variance = mad(diff, diff, variance);
    }
#endif

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
    barrier(CLK_LOCAL_MEM_FENCE);
    invVar = rsqrt(variance + epsilon);

#else // MIO_BN_USESAVED == 1

    mean   = lmean;
    invVar = lvar;

#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    ds             = 0.;
    db             = 0.;
    _FLOAT dyvalue = 0.;
    _FLOAT xhat    = 0.;

#pragma unroll
    for(unsigned int k = lid; k < MIO_BN_LESS; k += MIO_BN_GRP0)
    {
        nidx    = iDiv(k, MIO_BN_HW);
        hwidx   = iMod(k, nidx, MIO_BN_HW);
        index   = nidx * MIO_BN_CHW + chwid + hwidx;
        dyvalue = *(dy_in + index);
        xhat    = *(x_in + index);
        xhat -= mean;
        xhat *= invVar;
        db += dyvalue;
        ds = mad(xhat, dyvalue, ds);
    }

#if(MIO_BN_REM)
    nidx  = iDiv(MIO_BN_LESS + lid, MIO_BN_HW);
    hwidx = iMod(MIO_BN_LESS + lid, nidx, MIO_BN_HW);
    index = nidx * MIO_BN_CHW + chwid + hwidx;
    if(index < MIO_BN_NCHW)
    {
        dyvalue = *(dy_in + index);
        xhat    = *(x_in + index);
        xhat -= mean;
        xhat *= invVar;
        db += dyvalue;
        ds = mad(xhat, dyvalue, ds);
    }
#endif
    // printf("db: %f ds: %f\n", db, ds);
    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
//===dBias reduction=======================
#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&db, lcl_data, lid, 1);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&db, lcl_data, lid, 1);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    //===dScale reduction=======================
    lcl_data[lid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&ds, lcl_data, lid, 1);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&ds, lcl_data, lid, 1);
#endif

    //===========================================
    barrier(CLK_LOCAL_MEM_FENCE);
    pscale = lcl_scale;

#pragma unroll
    for(unsigned int k = lid; k < MIO_BN_LESS; k += MIO_BN_GRP0)
    {
        nidx    = iDiv(k, MIO_BN_HW);
        hwidx   = iMod(k, nidx, MIO_BN_HW);
        index   = nidx * MIO_BN_CHW + chwid + hwidx;
        dyvalue = *(dy_in + index);
        xhat    = *(x_in + index);
        xhat -= mean;
        xhat *= invVar;
        _FLOAT tmp1       = mad(NHW, dyvalue, -db);
        _FLOAT tmp2       = -xhat * ds;
        _FLOAT tmp3       = pscale * invVar * INHW;
        *(dx_out + index) = tmp3 * (tmp2 + tmp1);
    }

#if(MIO_BN_REM)
    nidx  = iDiv(MIO_BN_LESS + lid, MIO_BN_HW);
    hwidx = iMod(MIO_BN_LESS + lid, nidx, MIO_BN_HW);
    index = nidx * MIO_BN_CHW + chwid + hwidx;
    if(index < MIO_BN_NCHW)
    {
        dyvalue = *(dy_in + index);
        xhat    = *(x_in + index);
        xhat -= mean;
        xhat *= invVar;
        _FLOAT tmp1       = mad(NHW, dyvalue, -db);
        _FLOAT tmp2       = -xhat * ds;
        _FLOAT tmp3       = pscale * invVar * INHW;
        *(dx_out + index) = tmp3 * (tmp2 + tmp1);
    }
#endif

    if(lid == 0)
    {
        // printf("db: %f, ds: %f\n", db, ds);
        *(dbias + grpid)  = db;
        *(dscale + grpid) = ds;
    }
}

#elif(MIO_BN_VARIANT == 6)

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
BatchNormBwdSpatial(const __global _FLOAT* __restrict x_in,
                    const __global _FLOAT* __restrict dy_in,
                    __global _FLOAT* __restrict dx_out,
                    const __global _FLOAT* bnScale,
                    __global _FLOAT* __restrict dscale,
                    __global _FLOAT* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                    double epsilon,
#elif(MIO_BN_USESAVED == 1)
                    const __global _FLOAT* savedMean,
                    const __global _FLOAT* savedInvVariance,
#endif
                    float INHW)
{

    // SPATIAL
    _FLOAT mean   = 0.;
    _FLOAT invVar = 0.;
    _FLOAT pscale = 0.;
    _FLOAT db     = 0.;
    _FLOAT ds     = 0.;

#if(MIO_BN_USESAVED == 1)
    __local _FLOAT lmean, lvar;
#endif

    __local _FLOAT lcl_scale;
    __local _FLOAT lcl_data[MIO_BN_LDS_SIZE];

    _FLOAT NHW = (_FLOAT)MIO_BN_NHW;

    unsigned int index = 0;
    unsigned int lid   = get_local_id(0);
    unsigned int grpid = get_group_id(0);
    unsigned int chwid = grpid * MIO_BN_HW;

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

    _FLOAT variance = 0.;
    //==== CALC MEAN =======================
    mean = 0.;

#pragma unroll
    for(unsigned int n = 0; n < MIO_BN_N; n++)
    {
        for(unsigned int hw = lid; hw < MIO_BN_HW; hw += MIO_BN_GRP0)
        {

            index = n * MIO_BN_CHW + chwid + hw;
            mean += (index < MIO_BN_NCHW) ? *(x_in + index) : 0.;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
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
    _FLOAT diff = 0.;
    variance    = 0.;
#pragma unroll
    for(unsigned int n = 0; n < MIO_BN_N; n++)
    {
        for(unsigned int hw = lid; hw < MIO_BN_HW; hw += MIO_BN_GRP0)
        {
            index    = n * MIO_BN_CHW + chwid + hw;
            diff     = ((index < MIO_BN_NCHW) ? *(x_in + index) : 0.) - mean;
            variance = mad(diff, diff, variance);
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
    barrier(CLK_LOCAL_MEM_FENCE);
    invVar = rsqrt(variance + epsilon);

#else // MIO_BN_USESAVED == 1

    mean   = lmean;
    invVar = lvar;

#endif
    barrier(CLK_LOCAL_MEM_FENCE);
    ds             = 0.;
    db             = 0.;
    _FLOAT dyvalue = 0.;
    _FLOAT xhat    = 0.;

#pragma unroll
    for(unsigned int n = 0; n < MIO_BN_N; n++)
    {
        for(unsigned int hw = lid; hw < MIO_BN_HW; hw += MIO_BN_GRP0)
        {
            index = n * MIO_BN_CHW + chwid + hw;
            if(index < MIO_BN_NCHW)
            {
                dyvalue = *(dy_in + index);
                xhat    = *(x_in + index);
                xhat -= mean;
                xhat *= invVar;
                db += dyvalue;
                ds = mad(xhat, dyvalue, ds);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    lcl_data[lid] = db;
    barrier(CLK_LOCAL_MEM_FENCE);
//===dBias reduction=======================
#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&db, lcl_data, lid, 1);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&db, lcl_data, lid, 1);
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

    //===dScale reduction=======================
    lcl_data[lid] = ds;
    barrier(CLK_LOCAL_MEM_FENCE);
#ifdef __AMDGCN__
#pragma unroll
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 32; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    dppLDSReduce64(&ds, lcl_data, lid, 1);
#else
    for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
    {
        if(lid < red)
            lcl_data[lid] += lcl_data[lid + red];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    regLDSreduce(&ds, lcl_data, lid, 1);
#endif

    //===========================================
    barrier(CLK_LOCAL_MEM_FENCE);
    pscale = lcl_scale;

#pragma unroll
    for(unsigned int n = 0; n < MIO_BN_N; n++)
    {
        for(unsigned int hw = lid; hw < MIO_BN_HW; hw += MIO_BN_GRP0)
        {

            index = n * MIO_BN_CHW + chwid + hw;
            if(index < MIO_BN_NCHW)
            {
                dyvalue = *(dy_in + index);
                xhat    = *(x_in + index);
                xhat -= mean;
                xhat *= invVar;
                _FLOAT tmp1       = mad(NHW, dyvalue, -db);
                _FLOAT tmp2       = -xhat * ds;
                _FLOAT tmp3       = pscale * invVar * INHW;
                *(dx_out + index) = tmp3 * (tmp2 + tmp1);
            }
        }
    }

    if(lid == 0)
    {
        // printf("db: %f, ds: %f\n", db, ds);
        *(dbias + grpid)  = db;
        *(dscale + grpid) = ds;
    }
}

#endif // END VARIANTS

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
