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
    _FLOAT_ACCUM ds         = (_FLOAT_ACCUM)0.;
    _FLOAT_ACCUM db         = (_FLOAT_ACCUM)0.;

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
            batchvalues[n] = FLOAT2FLOATPREC(*(x_in + index));
            mean += batchvalues[n];
            variance = mad(batchvalues[n], batchvalues[n], variance);
        }
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        batchvalues[MIO_BN_NLOOPM] =
            (index < MIO_BN_NCHW) ? FLOAT2FLOATPREC(*(x_in + index)) : (_FLOAT_PREC)0.;
        mean += batchvalues[MIO_BN_NLOOPM];
        variance = mad(batchvalues[MIO_BN_NLOOPM], batchvalues[MIO_BN_NLOOPM], variance);
    }

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
    invVariance = rsqrt(variance + epsilon);
#endif // end -- Recalc mean and variance
    //-------------------------------------------

    //==== CALC DB and DS =========================================
    if(lid < MIO_BN_SEGMENT)
    {

        for(unsigned int n = 0; n < MIO_BN_NLOOPM; ++n)
        {
            nid         = n * MIO_BN_SEGIHW + lidihw;
            index       = nid * MIO_BN_CHW + chwid;
            dyvalues[n] = FLOAT2FLOATPREC(*(dy_in + index));
            db += dyvalues[n];

#if(MIO_BN_USESAVED == 1)
            batchvalues[n] = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
#else
            batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif
            // batchvalues is now xhat
            ds = mad(batchvalues[n], dyvalues[n], ds);
        }
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        dyvalues[MIO_BN_NLOOPM] =
            ((index < MIO_BN_NCHW) ? FLOAT2FLOATPREC(*(dy_in + index)) : (_FLOAT_PREC)0.);
        db += dyvalues[MIO_BN_NLOOPM];

#if(MIO_BN_USESAVED == 1)
        batchvalues[MIO_BN_NLOOPM] = (index < MIO_BN_NCHW)
                                         ? ((FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance)
                                         : (_FLOAT_PREC)0.;

#else
        batchvalues[MIO_BN_NLOOPM] = (batchvalues[MIO_BN_NLOOPM] - mean) * invVariance;
#endif
        // batchvalues is now xhat
        ds = mad(batchvalues[MIO_BN_NLOOPM], dyvalues[MIO_BN_NLOOPM], ds);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDS_SIZE];
    lds_reduce2(&ds, &db, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, lid);
#else
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&ds, &db, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, lid);
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
            dx_out[index] = FLOATPREC2FLOAT(tmp3 * (tmp2 + tmp1));
        } // end for
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        if(index < MIO_BN_NCHW)
        {
            tmp1          = mad(NHW, dyvalues[MIO_BN_NLOOPM], -db);
            tmp2          = -batchvalues[MIO_BN_NLOOPM] * ds;
            tmp3          = (pscale * invVariance) * INHW;
            dx_out[index] = FLOATPREC2FLOAT(tmp3 * (tmp2 + tmp1));
        }
    }
    if(lid == 0)
    {
        dbias[grpid]  = (_FLOAT_PREC)db;
        dscale[grpid] = (_FLOAT_PREC)ds;
    }
} // end spatial

#elif(MIO_BN_VARIANT == 1)

#if MIO_LAYOUT_NHWC
#define MIO_MAX_READ 1
#define RD_BLK 1
#define GRPRD (MIO_BN_GRP0 * RD_BLK)
#else
#define MIO_MAX_READ 2
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
    _FLOAT_ACCUM db         = (_FLOAT_ACCUM)0.;
    _FLOAT_ACCUM ds         = (_FLOAT_ACCUM)0.;
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
#if !MIO_LAYOUT_NHWC
    unsigned int chwid = grpid * MIO_BN_HW;
#endif
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
#if !MIO_LAYOUT_NHWC && MIO_BN_HW >= 4096
    _FLOAT4 read4;
#if(MIO_BN_N > MIO_BN_LOOP_UNROLL_MAXN)
    __attribute__((opencl_unroll_hint(4))) for(unsigned int k = lid << 2; k < MIO_BN_LESS4;
                                               k += GRPRD)
#else
    __attribute__((opencl_unroll_hint(2))) for(unsigned int k = lid << 2; k < MIO_BN_LESS4;
                                               k += GRPRD)
#endif
    {
        nidx  = k / MIO_BN_HW;
        hwidx = k - (nidx * MIO_BN_HW);
        index = nidx * MIO_BN_CHW + chwid + hwidx;
        read4 = *((const global _FLOAT4*)(x_in + index));
        mean += FLOAT2FLOATPREC(read4.x);
        mean += FLOAT2FLOATPREC(read4.y);
        mean += FLOAT2FLOATPREC(read4.z);
        mean += FLOAT2FLOATPREC(read4.w);
        variance = mad(FLOAT2FLOATPREC(read4.x), FLOAT2FLOATPREC(read4.x), variance);
        variance = mad(FLOAT2FLOATPREC(read4.y), FLOAT2FLOATPREC(read4.y), variance);
        variance = mad(FLOAT2FLOATPREC(read4.z), FLOAT2FLOATPREC(read4.z), variance);
        variance = mad(FLOAT2FLOATPREC(read4.w), FLOAT2FLOATPREC(read4.w), variance);
    }

#if(MIO_BN_REM4)
    if(lid < MIO_BN_REM4)
    {
        unsigned int remkey = lid + MIO_BN_LESS4;
        nidx                = remkey / MIO_BN_HW;
        hwidx               = remkey - (nidx * MIO_BN_HW);
        index               = nidx * MIO_BN_CHW + chwid + hwidx;
        if(index < (MIO_BN_NCHW - 3))
        {
            read4 = *((const global _FLOAT4*)(x_in + index));
            mean += FLOAT2FLOATPREC(read4.x);
            mean += FLOAT2FLOATPREC(read4.y);
            mean += FLOAT2FLOATPREC(read4.z);
            mean += FLOAT2FLOATPREC(read4.w);
            variance = mad(FLOAT2FLOATPREC(read4.x), FLOAT2FLOATPREC(read4.x), variance);
            variance = mad(FLOAT2FLOATPREC(read4.y), FLOAT2FLOATPREC(read4.y), variance);
            variance = mad(FLOAT2FLOATPREC(read4.z), FLOAT2FLOATPREC(read4.z), variance);
            variance = mad(FLOAT2FLOATPREC(read4.w), FLOAT2FLOATPREC(read4.w), variance);
        }
    }
#endif
#else
#if(MIO_BN_N > MIO_BN_LOOP_UNROLL_MAXN)
    __attribute__((opencl_unroll_hint(4))) for(unsigned int k = lid; k < MIO_BN_LESS;
                                               k += MIO_BN_GRP0)
#else
    for(unsigned int k = lid; k < MIO_BN_LESS; k += MIO_BN_GRP0)
#endif
    {
        nidx           = k / MIO_BN_HW;
        hwidx          = k - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
        index          = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
#else
        index = nidx * MIO_BN_CHW + chwid + hwidx;
#endif
        _FLOAT_PREC in = FLOAT2FLOATPREC(*(x_in + index));
        mean += in;
        variance = mad(in, in, variance);
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
        _FLOAT_PREC in = (index < MIO_BN_NCHW) ? FLOAT2FLOATPREC(*(x_in + index)) : (_FLOAT_PREC)0.;
        mean += in;
        variance = mad(in, in, variance);
    }
#endif // end REM
#endif // end if 4096

    barrier(CLK_LOCAL_MEM_FENCE);
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
    variance = mad(-mean, mean, variance);
    if(variance < 0)
    {
        variance = 0;
    }
    invVariance = rsqrt(variance + epsilon);

#else // MIO_BN_USESAVED == 1

    mean        = lmean;
    invVariance = lvar;

#endif

#if MIO_LAYOUT_NHWC
    _FLOAT dyRead;
    _FLOAT xread;
    _FLOAT_PREC xhat_tmp;
#else
    _FLOAT4 dyRead4;
    _FLOAT4 xread4;
    _FLOAT_PREC4 xhat4;
#endif
#if(MIO_BN_N > MIO_BN_LOOP_UNROLL_MAXN)
    __attribute__((opencl_unroll_hint(4))) for(unsigned int k = lid << 2 * (1 - MIO_LAYOUT_NHWC);
                                               k < MIO_BN_LESS4;
                                               k += GRPRD)
#else
    __attribute__((opencl_unroll_hint(2))) for(unsigned int k = lid << 2 * (1 - MIO_LAYOUT_NHWC);
                                               k < MIO_BN_LESS4;
                                               k += GRPRD)
#endif
    {
        nidx     = k / MIO_BN_HW;
        hwidx    = k - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
        index    = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
        xread    = *((const global _FLOAT*)(x_in + index));
        dyRead   = *((const global _FLOAT*)(dy_in + index));
        xhat_tmp = (FLOAT2FLOATPREC(xread) - mean) * invVariance;
        db += FLOAT2FLOATPREC(dyRead);
        ds = mad(xhat_tmp, FLOAT2FLOATPREC(dyRead), ds);
#else
        index   = nidx * MIO_BN_CHW + chwid + hwidx;
        xread4  = *((const global _FLOAT4*)(x_in + index));
        dyRead4 = *((const global _FLOAT4*)(dy_in + index));
        xhat4.x = (FLOAT2FLOATPREC(xread4.x) - mean) * invVariance;
        xhat4.y = (FLOAT2FLOATPREC(xread4.y) - mean) * invVariance;
        xhat4.z = (FLOAT2FLOATPREC(xread4.z) - mean) * invVariance;
        xhat4.w = (FLOAT2FLOATPREC(xread4.w) - mean) * invVariance;
        db += FLOAT2FLOATPREC(dyRead4.x);
        db += FLOAT2FLOATPREC(dyRead4.y);
        db += FLOAT2FLOATPREC(dyRead4.z);
        db += FLOAT2FLOATPREC(dyRead4.w);
        ds = mad(xhat4.x, FLOAT2FLOATPREC(dyRead4.x), ds);
        ds = mad(xhat4.y, FLOAT2FLOATPREC(dyRead4.y), ds);
        ds = mad(xhat4.z, FLOAT2FLOATPREC(dyRead4.z), ds);
        ds = mad(xhat4.w, FLOAT2FLOATPREC(dyRead4.w), ds);
#endif
    }

#if(MIO_BN_REM4)
    unsigned int remkey = (lid << 2 * (1 - MIO_LAYOUT_NHWC)) + MIO_BN_LESS4;
    nidx                = remkey / MIO_BN_HW;
    hwidx               = remkey - (nidx * MIO_BN_HW);
    index               = nidx * MIO_BN_CHW +
#if MIO_LAYOUT_NHWC
            hwidx * MIO_BN_C + grpid;
    if(index < MIO_BN_NCHW)
    {
        xread    = *((const global _FLOAT*)(x_in + index));
        dyRead   = *((const global _FLOAT*)(dy_in + index));
        xhat_tmp = (FLOAT2FLOATPREC(xread) - mean) * invVariance;
        db += FLOAT2FLOATPREC(dyRead);
        ds = mad(xhat_tmp, FLOAT2FLOATPREC(dyRead), ds);
#else
            chwid + hwidx;
    if(index < (MIO_BN_NCHW - 3))
    {
        xread4  = *((const global _FLOAT4*)(x_in + index));
        dyRead4 = *((const global _FLOAT4*)(dy_in + index));
        xhat4.x = (FLOAT2FLOATPREC(xread4.x) - mean) * invVariance;
        xhat4.y = (FLOAT2FLOATPREC(xread4.y) - mean) * invVariance;
        xhat4.z = (FLOAT2FLOATPREC(xread4.z) - mean) * invVariance;
        xhat4.w = (FLOAT2FLOATPREC(xread4.w) - mean) * invVariance;
        db += FLOAT2FLOATPREC(dyRead4.x);
        db += FLOAT2FLOATPREC(dyRead4.y);
        db += FLOAT2FLOATPREC(dyRead4.z);
        db += FLOAT2FLOATPREC(dyRead4.w);
        ds = mad(xhat4.x, FLOAT2FLOATPREC(dyRead4.x), ds);
        ds = mad(xhat4.y, FLOAT2FLOATPREC(dyRead4.y), ds);
        ds = mad(xhat4.z, FLOAT2FLOATPREC(dyRead4.z), ds);
        ds = mad(xhat4.w, FLOAT2FLOATPREC(dyRead4.w), ds);
#endif
    }

#endif
    barrier(CLK_GLOBAL_MEM_FENCE);

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDS_SIZE];
    lds_reduce2(&ds, &db, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, lid);
#else
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&ds, &db, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, lid);
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
        *(dbias + grpid)  = (_FLOAT_PREC)db;
        *(dscale + grpid) = (_FLOAT_PREC)ds;
#endif
    }

    _FLOAT_PREC vals[MIO_MAX_READ];
#if(MIO_BN_N > MIO_BN_LOOP_UNROLL_MAXN)
    __attribute__((opencl_unroll_hint(4))) for(unsigned int k = (MIO_MAX_READ * lid);
                                               k < MIO_BN_LESSOUT;
                                               k += MIO_BN_CHUNK)
    {
        __attribute__((opencl_unroll_hint(4))) for(unsigned int j = 0; j < MIO_MAX_READ; j++)
#else
    for(unsigned int k = (MIO_MAX_READ * lid); k < MIO_BN_LESSOUT; k += MIO_BN_CHUNK)
    {
        for(unsigned int j = 0; j < MIO_MAX_READ; j++)
#endif
        {
            unsigned int l  = k + j;
            nidx            = l / MIO_BN_HW;
            hwidx           = l - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
            index           = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
#else
            index   = nidx * MIO_BN_CHW + chwid + hwidx;
#endif
            dyvalue         = FLOAT2FLOATPREC(*(dy_in + index));
            xhat            = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
#if MIOPEN_USE_FP16 == 1
            float temp_tmp1 = mad((float)NHW, (float)dyvalue, -temp_db);
            float temp_tmp2 = -((float)xhat) * temp_ds;
            float temp_vals = (float)tmp3 * (temp_tmp2 + temp_tmp1);
            vals[j]         = (_FLOAT_PREC)temp_vals;
#else
            tmp1    = mad(NHW, dyvalue, -db);
            tmp2    = -xhat * ds;
            vals[j] = tmp3 * (tmp2 + tmp1);
#endif
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
#if(MIO_BN_N > MIO_BN_LOOP_UNROLL_MAXN)
        __attribute__((opencl_unroll_hint(4))) for(unsigned int j = 0; j < MIO_MAX_READ; j++)
#else
        for(unsigned int j = 0; j < MIO_MAX_READ; j++)
#endif
        {
            unsigned int l    = k + j;
            nidx              = l / MIO_BN_HW;
            hwidx             = l - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
            index             = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
#else
            index = nidx * MIO_BN_CHW + chwid + hwidx;
#endif
            *(dx_out + index) = FLOATPREC2FLOAT(vals[j]);
        }
    }

#if(MIO_BN_REMOUT)
    unsigned int remkeyout = (MIO_MAX_READ * lid) + MIO_BN_LESSOUT;
#if(MIO_BN_N > MIO_BN_LOOP_UNROLL_MAXN)
    __attribute__((opencl_unroll_hint(4))) for(unsigned int j = 0; j < MIO_MAX_READ; j++)
#else
    for(unsigned int j = 0; j < MIO_MAX_READ; j++)
#endif
    {
        unsigned int l = remkeyout + j;
        nidx           = l / MIO_BN_HW;
        hwidx          = l - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
        index          = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
#else
        index = nidx * MIO_BN_CHW + chwid + hwidx;
#endif
        if(index < MIO_BN_NCHW)
        {
            dyvalue = FLOAT2FLOATPREC(*(dy_in + index));
            xhat    = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
            tmp1    = mad(NHW, dyvalue, -db);
            tmp2    = -xhat * ds;
            vals[j] = tmp3 * (tmp2 + tmp1);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
#if(MIO_BN_N > MIO_BN_LOOP_UNROLL_MAXN)
    __attribute__((opencl_unroll_hint(4))) for(unsigned int j = 0; j < MIO_MAX_READ; j++)
#else
    for(unsigned int j = 0; j < MIO_MAX_READ; j++)
#endif
    {
        unsigned int l = remkeyout + j;
        nidx           = l / MIO_BN_HW;
        hwidx          = l - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
        index          = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
#else
        index = nidx * MIO_BN_CHW + chwid + hwidx;
#endif
        if(index < MIO_BN_NCHW)
        {
            *(dx_out + index) = FLOATPREC2FLOAT(vals[j]);
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

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDS_SIZE];
    lds_reduce2(&mean, &variance, FLOAT2ACCUM(INHW), lcl_data_x, lcl_data_y, lid);
#else
    commitID = 64;
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, FLOAT2ACCUM(INHW), lcl_data_x, lcl_data_y, lid);
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

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_NGRPS];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_NGRPS];
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
#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDS_SIZE];
    lds_reduce2(&dscale, &dbias, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, ylid);
#else
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&dscale, &dbias, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, ylid);
#endif

    // end reduction-----------
    if(ylid == 0)
    {
        unsigned int betaindex  = cidx + ygrp_sz * ygrp_id + 6;
        unsigned int gammaindex = cidx + ygrp_sz * ygrp_id + 4;
        buff[gammaindex]        = FLOAT2FLOATPREC(dscale);
        buff[betaindex]         = FLOAT2FLOATPREC(dbias);
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

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_NGRPS];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_NGRPS];
    lds_reduce2(&ds, &db, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, lid);
#else
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&ds, &db, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, lid);
#endif

    if(ygid == 0)
    {
        delta_scale[xgid] = FLOAT2FLOATPREC(ds);
        delta_bias[xgid]  = FLOAT2FLOATPREC(db);
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
            mean += batchvalues[n] = FLOAT2FLOATPREC(*(x_in + index));
            variance               = mad(batchvalues[n], batchvalues[n], variance);
#else
            _FLOAT_PREC in = FLOAT2FLOATPREC(*(x_in + index));
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
#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDS_SIZE];
    lds_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#else
    local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
#endif

    // REDUCTION COMPLETE -----------------------
    variance = mad(-mean, mean, variance);
    if(variance < 0)
    {
        variance = 0;
    }
    invVariance = rsqrt(variance + epsilon);

// RECALC of MEAN and VARIANCE complete
//===========================================
#endif

    if(lid < MIO_BN_HW)
    {
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index             = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            db += dyvalues[n] = FLOAT2FLOATPREC(*(dy_in + index));

#if(MIO_BN_USESAVED == 1)
            batchvalues[n]    = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
#else
            batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif // batchvalues is now xhat

            ds = mad(batchvalues[n], dyvalues[n], ds);
#else  // maxn
            db += FLOAT2FLOATPREC(*(dy_in + index));
            _FLOAT_PREC xhat = ((FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance);
            ds               = mad(xhat, FLOAT2FLOATPREC(*(dy_in + index)), ds);
#endif
        }
    }
    else
    {
        db = (_FLOAT_PREC)0.;
        ds = (_FLOAT_PREC)0.;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if !MIOPEN_USE_AMDGCN
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDS_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDS_SIZE];
    lds_reduce2(&ds, &db, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, lid);
#else
    local _FLOAT_ACCUM lcl_data_x2[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM lcl_data_y2[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&ds, &db, (_FLOAT_ACCUM)1.0, lcl_data_x2, lcl_data_y2, lid);
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
            tmp1 = mad(NHW, FLOAT2FLOATPREC(*(dy_in + index)), -db);
            tmp2 = -(FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance * ds;
#endif
            tmp3          = (pscale * invVariance) * INHW;
            dx_out[index] = FLOATPREC2FLOAT(tmp3 * (tmp2 + tmp1));
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
