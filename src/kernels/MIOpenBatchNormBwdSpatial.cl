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
#if defined(__AMDGCN__) && !(MIO_BN_GFX103X || MIO_BN_GFX110X || MIO_BN_GFX120X || MIO_BN_GFX115X)
#undef MIOPEN_USE_AMDGCN
#define MIOPEN_USE_AMDGCN 1
#endif

#include "batchnorm_functions.h"
#include "bnorm_spatial_activation_functions.h"
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
                          const __global _FLOAT_PREC* __restrict bnScale,
                          const __global _FLOAT_PREC* __restrict bnBias,
                          __global _FLOAT_PREC* __restrict dscale,
                          __global _FLOAT_PREC* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                          double epsilon,
#elif(MIO_BN_USESAVED == 1)
                          const __global _FLOAT_PREC* savedMean,
                          const __global _FLOAT_PREC* savedInvVariance,
#endif
                          _FLOAT_PREC INHW,
                          _FLOAT_PREC _alpha,
                          _FLOAT_PREC _beta)
{

    ACTIVATION_SET()
    // SPATIAL
    _FLOAT_PREC mean = (_FLOAT_PREC)0.;
#if(MIO_BN_USESAVED == 0)
    _FLOAT_PREC variance = (_FLOAT_PREC)0.;
#endif
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pscale      = (_FLOAT_PREC)0.;
    _FLOAT_PREC pbias       = (_FLOAT_PREC)0.;
    _FLOAT_ACCUM ds         = (_FLOAT_ACCUM)0.;
    _FLOAT_ACCUM db         = (_FLOAT_ACCUM)0.;

    _FLOAT_PREC batchvalues[MIO_BN_NLOOP];
    _FLOAT_PREC dyvalues[MIO_BN_NLOOP];

    __local _FLOAT_PREC lbns;
#if(MIOPEN_NRN_OP_ID > 0)
    __local _FLOAT_PREC lbnb;
#endif

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
#if(MIOPEN_NRN_OP_ID > 0)
        lbnb = *(bnBias + grpid);
#endif

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
    pscale = lbns;
#if(MIOPEN_NRN_OP_ID > 0)
    pbias = lbnb;
#endif

    //==== CALC DB and DS =========================================
    if(lid < MIO_BN_SEGMENT)
    {
        for(unsigned int n = 0; n < MIO_BN_NLOOPM; ++n)
        {
            nid         = n * MIO_BN_SEGIHW + lidihw;
            index       = nid * MIO_BN_CHW + chwid;
            dyvalues[n] = FLOAT2FLOATPREC(*(dy_in + index));

#if(MIO_BN_USESAVED == 1)
            batchvalues[n] = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
#else
            batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif
            ACTIVATION_OP_BWD(dyvalues[n], batchvalues[n], pscale, pbias, dyvalues[n], _FLOAT_PREC)
            // batchvalues is now xhat
            db += dyvalues[n];
            ds = mad(batchvalues[n], dyvalues[n], ds);
        }
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        dyvalues[MIO_BN_NLOOPM] =
            ((index < MIO_BN_NCHW) ? FLOAT2FLOATPREC(*(dy_in + index)) : (_FLOAT_PREC)0.);

#if(MIO_BN_USESAVED == 1)
        batchvalues[MIO_BN_NLOOPM] = (index < MIO_BN_NCHW)
                                         ? ((FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance)
                                         : (_FLOAT_PREC)0.;
#else
        batchvalues[MIO_BN_NLOOPM] = (batchvalues[MIO_BN_NLOOPM] - mean) * invVariance;
#endif
        ACTIVATION_OP_BWD(dyvalues[MIO_BN_NLOOPM],
                          batchvalues[MIO_BN_NLOOPM],
                          pscale,
                          pbias,
                          dyvalues[MIO_BN_NLOOPM],
                          _FLOAT_PREC)
        // batchvalues is now xhat
        db += dyvalues[MIO_BN_NLOOPM];
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
        _FLOAT_PREC value;
        for(unsigned int n = 0; n < MIO_BN_NLOOPM; n++)
        { // apply normalization
            nid           = n * MIO_BN_SEGIHW + lidihw;
            index         = nid * MIO_BN_CHW + chwid;
            tmp1          = mad(NHW, dyvalues[n], -db);
            tmp2          = -batchvalues[n] * ds;
            tmp3          = (pscale * invVariance) * INHW;
            value         = tmp3 * (tmp2 + tmp1);
            dx_out[index] = FLOATPREC2FLOAT(value);
        } // end for
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        if(index < MIO_BN_NCHW)
        {
            tmp1          = mad(NHW, dyvalues[MIO_BN_NLOOPM], -db);
            tmp2          = -batchvalues[MIO_BN_NLOOPM] * ds;
            tmp3          = (pscale * invVariance) * INHW;
            value         = tmp3 * (tmp2 + tmp1);
            dx_out[index] = FLOATPREC2FLOAT(value);
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
                          const __global _FLOAT_PREC* __restrict bnScale,
                          const __global _FLOAT_PREC* __restrict bnBias,
                          __global _FLOAT_PREC* __restrict dscale,
                          __global _FLOAT_PREC* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                          double epsilon,
#elif(MIO_BN_USESAVED == 1)
                          const __global _FLOAT_PREC* savedMean,
                          const __global _FLOAT_PREC* savedInvVariance,
#endif
                          _FLOAT_PREC INHW,
                          _FLOAT_PREC _alpha,
                          _FLOAT_PREC _beta)
{

    ACTIVATION_SET()
    // SPATIAL
    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pscale      = (_FLOAT_PREC)0.;
    _FLOAT_PREC pbias       = (_FLOAT_PREC)0.;
    _FLOAT_ACCUM db         = (_FLOAT_ACCUM)0.;
    _FLOAT_ACCUM ds         = (_FLOAT_ACCUM)0.;
    _FLOAT_PREC xhat        = (_FLOAT_PREC)0.;

#if(MIO_BN_USESAVED == 1)
    __local _FLOAT_PREC lmean, lvar;
#endif

    __local _FLOAT_PREC lcl_scale;
#if(MIOPEN_NRN_OP_ID > 0)
    __local _FLOAT_PREC lcl_bias;
#endif
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
#if(MIOPEN_NRN_OP_ID > 0)
        lcl_bias  = *(bnBias + grpid);
#endif
#if(MIO_BN_USESAVED == 1)
        lmean     = *(savedMean + grpid);
        lvar      = *(savedInvVariance + grpid);
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    pscale = lcl_scale;
#if(MIOPEN_NRN_OP_ID > 0)
    pbias  = lcl_bias;
#endif

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
    _FLOAT_PREC dyvalue;
    _FLOAT_PREC xhat_tmp;
#else
    _FLOAT4 dyRead4;
    _FLOAT4 xread4;
    _FLOAT_PREC4 dyvalue4;
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
        dyvalue  = FLOAT2FLOATPREC(dyRead);
        xhat_tmp = (FLOAT2FLOATPREC(xread) - mean) * invVariance;
        ACTIVATION_OP_BWD(dyvalue, xhat_tmp, pscale, pbias, dyvalue, _FLOAT_PREC)
        db += dyvalue;
        ds = mad(xhat_tmp, dyvalue, ds);
#else
        index      = nidx * MIO_BN_CHW + chwid + hwidx;
        xread4     = *((const global _FLOAT4*)(x_in + index));
        dyRead4    = *((const global _FLOAT4*)(dy_in + index));
        dyvalue4.x = FLOAT2FLOATPREC(dyRead4.x);
        dyvalue4.y = FLOAT2FLOATPREC(dyRead4.y);
        dyvalue4.z = FLOAT2FLOATPREC(dyRead4.z);
        dyvalue4.w = FLOAT2FLOATPREC(dyRead4.w);
        xhat4.x    = (FLOAT2FLOATPREC(xread4.x) - mean) * invVariance;
        xhat4.y    = (FLOAT2FLOATPREC(xread4.y) - mean) * invVariance;
        xhat4.z    = (FLOAT2FLOATPREC(xread4.z) - mean) * invVariance;
        xhat4.w    = (FLOAT2FLOATPREC(xread4.w) - mean) * invVariance;

        ACTIVATION_OP_BWD(dyvalue4, xhat4, pscale, pbias, dyvalue4, _FLOAT_PREC4)

        db += dyvalue4.x;
        db += dyvalue4.y;
        db += dyvalue4.z;
        db += dyvalue4.w;
        ds = mad(xhat4.x, dyvalue4.x, ds);
        ds = mad(xhat4.y, dyvalue4.y, ds);
        ds = mad(xhat4.z, dyvalue4.z, ds);
        ds = mad(xhat4.w, dyvalue4.w, ds);
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
        dyvalue  = FLOAT2FLOATPREC(dyRead);
        xhat_tmp = (FLOAT2FLOATPREC(xread) - mean) * invVariance;
        ACTIVATION_OP_BWD(dyvalue, xhat_tmp, pscale, pbias, dyvalue, _FLOAT_PREC)
        db += dyvalue;
        ds = mad(xhat_tmp, dyvalue, ds);
#else
            chwid + hwidx;
    if(index < (MIO_BN_NCHW - 3))
    {
        xread4     = *((const global _FLOAT4*)(x_in + index));
        dyRead4    = *((const global _FLOAT4*)(dy_in + index));
        dyvalue4.x = FLOAT2FLOATPREC(dyRead4.x);
        dyvalue4.y = FLOAT2FLOATPREC(dyRead4.y);
        dyvalue4.z = FLOAT2FLOATPREC(dyRead4.z);
        dyvalue4.w = FLOAT2FLOATPREC(dyRead4.w);

        xhat4.x = (FLOAT2FLOATPREC(xread4.x) - mean) * invVariance;
        xhat4.y = (FLOAT2FLOATPREC(xread4.y) - mean) * invVariance;
        xhat4.z = (FLOAT2FLOATPREC(xread4.z) - mean) * invVariance;
        xhat4.w = (FLOAT2FLOATPREC(xread4.w) - mean) * invVariance;

        ACTIVATION_OP_BWD(dyvalue4, xhat4, pscale, pbias, dyvalue4, _FLOAT_PREC4)
        db += dyvalue4.x;
        db += dyvalue4.y;
        db += dyvalue4.z;
        db += dyvalue4.w;
        ds = mad(xhat4.x, dyvalue4.x, ds);
        ds = mad(xhat4.y, dyvalue4.y, ds);
        ds = mad(xhat4.z, dyvalue4.z, ds);
        ds = mad(xhat4.w, dyvalue4.w, ds);
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
    _FLOAT_PREC value1;
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
            unsigned int l = k + j;
            nidx           = l / MIO_BN_HW;
            hwidx          = l - (nidx * MIO_BN_HW);
#if MIO_LAYOUT_NHWC
            index          = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
#else
            index   = nidx * MIO_BN_CHW + chwid + hwidx;
#endif
            value1         = FLOAT2FLOATPREC(*(dy_in + index));
            xhat           = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
            ACTIVATION_OP_BWD(value1, xhat, pscale, pbias, value1, _FLOAT_PREC)
#if MIOPEN_USE_FP16 == 1
            float temp_tmp1 = mad((float)NHW, (float)value1, -temp_db);
            float temp_tmp2 = -((float)xhat) * temp_ds;
            float temp_vals = (float)tmp3 * (temp_tmp2 + temp_tmp1);
            vals[j]         = (_FLOAT_PREC)temp_vals;
#else
            tmp1    = mad(NHW, value1, -db);
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
            value1 = FLOAT2FLOATPREC(*(dy_in + index));
            xhat   = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
            ACTIVATION_OP_BWD(value1, xhat, pscale, pbias, value1, _FLOAT_PREC)
            tmp1    = mad(NHW, value1, -db);
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

__attribute__((reqd_work_group_size(MIO_BN_GRP0_FINAL, MIO_BN_GRP1_FINAL, MIO_BN_GRP2_FINAL)))
__kernel void
MIOpenBatchNormBwdSpatialFinalMeanVariance(__global _FLOAT* __restrict meanvarbuff,
                                           _FLOAT_PREC INHW,
                                           double epsilon)
{

    unsigned int xlid    = get_local_id(0);
    unsigned int ylid    = get_local_id(1);
    unsigned int zlid    = get_local_id(2);
    unsigned int xgrp_id = get_group_id(0);
    unsigned int xgid    = get_global_id(0);
    unsigned int xgrp_sz = get_local_size(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int zgrp_sz = get_local_size(2);

    unsigned int xstride = MIO_LAYOUT_NHWC ? 1 : MIO_BN_HW;
    unsigned int ystride = MIO_LAYOUT_NHWC ? MIO_BN_C : 1;

    if(xgid * VEC_SIZE_X >= MIO_BN_C)
        return;

    _FLOAT_PREC_C variance = (_FLOAT_PREC_C)0.;
    _FLOAT_PREC_C mean     = (_FLOAT_PREC_C)0.;
    _FLOAT_PREC_C invVariance;

    for(unsigned int zoffset = zlid; zoffset < MIO_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < MIO_BN_NGRPS; yoffset += ygrp_sz)
        {
            mean += loadFromStash((__global _FLOAT_C*)meanvarbuff,
                                  0,
                                  MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                                  MIO_BN_GRP1 * yoffset * VEC_SIZE_Y,
                                  ystride / VEC_SIZE_X,
                                  xgrp_sz,
                                  xgrp_id,
                                  xlid,
                                  xstride);
            variance += loadFromStash((__global _FLOAT_C*)meanvarbuff,
                                      1,
                                      MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                                      MIO_BN_GRP1 * yoffset * VEC_SIZE_Y,
                                      ystride / VEC_SIZE_X,
                                      xgrp_sz,
                                      xgrp_id,
                                      xlid,
                                      xstride);
        }
    }

#if !MIOPEN_USE_AMDGCN || MIO_BN_GRP0 > 1 || MIO_BN_LDSGCN_SIZE == 1 || VEC_SIZE_X > 1
    local _FLOAT_ACCUM_C lcl_data[2 * MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL];
    lds_reduce2_2d(
        &mean, &variance, INHW, lcl_data, xgrp_sz, xlid, ylid + zlid * ygrp_sz, ygrp_sz * zgrp_sz);
#else
    local _FLOAT_ACCUM_C lcl_data_x[MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL / 64];
    local _FLOAT_ACCUM_C lcl_data_y[MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL / 64];
    gcn_reduce2(&mean, &variance, INHW, lcl_data_x, lcl_data_y, ylid + zlid * ygrp_sz);
#endif

    variance    = mad(-mean, mean, variance);
    variance    = max(variance, (_FLOAT_PREC_C)0.);
    invVariance = rsqrt(variance + (_FLOAT_PREC_C)epsilon);

    for(unsigned int zoffset = zlid; zoffset < MIO_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < MIO_BN_NGRPS; yoffset += ygrp_sz)
        {
            // Replicate mean and variance for all y groups because stash == dx_out and
            // MIOpenBatchNormBwdSpatialDX will read them and rewrite the buffer entirely.
            storeToStash(mean,
                         (__global _FLOAT_C*)meanvarbuff,
                         0,
                         MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                         MIO_BN_GRP1 * yoffset * VEC_SIZE_Y,
                         ystride / VEC_SIZE_X,
                         xgrp_sz,
                         xgrp_id,
                         xlid,
                         xstride);
            storeToStash(invVariance,
                         (__global _FLOAT_C*)meanvarbuff,
                         1,
                         MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                         MIO_BN_GRP1 * yoffset * VEC_SIZE_Y,
                         ystride / VEC_SIZE_X,
                         xgrp_sz,
                         xgrp_id,
                         xlid,
                         xstride);
        }
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormBwdSpatialMeanVariance(const __global _FLOAT* __restrict in,
                                      __global _FLOAT* __restrict meanvarbuff)
{

    unsigned int xlid    = get_local_id(0);
    unsigned int ylid    = get_local_id(1);
    unsigned int zlid    = get_local_id(2);
    unsigned int xgrp_id = get_group_id(0);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int zgrp_id = get_group_id(2);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int zgid    = get_global_id(2);
    unsigned int xgrp_sz = get_local_size(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int zgrp_sz = get_local_size(2);

    unsigned int xstride = MIO_LAYOUT_NHWC ? 1 : MIO_BN_HW;
    unsigned int ystride = MIO_LAYOUT_NHWC ? MIO_BN_C : 1;

    if(xgid * VEC_SIZE_X >= MIO_BN_C)
        return;

    unsigned int index;
    _FLOAT_PREC_LS value;
    _FLOAT_PREC_C mean     = (_FLOAT_PREC_C)0.;
    _FLOAT_PREC_C variance = (_FLOAT_PREC_C)0.;

    if(ygid * VEC_SIZE_Y < MIO_BN_HW && zgid < MIO_BN_N)
    {
        _FLOAT_LS read4;
        unsigned int index_base = zgid * MIO_BN_N_ELEMENTS * MIO_BN_CHW +
                                  ygid * ystride * VEC_SIZE_Y + xgid * xstride * VEC_SIZE_X;
        for(unsigned int n = 0; n < MIO_BN_N_ELEMENTS; n++)
        {
            index = index_base + n * MIO_BN_CHW;
            read4 = *((const __global _FLOAT_LS*)(in + index));
            value = FLOAT2FLOATPREC_VEC(read4);
            _ACCUMULATE(mean, value)
            _ACCUMULATE_MAD(variance, value, value, variance)
        }
    }

#if !MIOPEN_USE_AMDGCN || MIO_BN_GRP0 > 1 || MIO_BN_LDSGCN_SIZE == 1 || VEC_SIZE_X > 1
    local _FLOAT_ACCUM_C lcl_data[2 * MIO_BN_LDS_SIZE];
    lds_reduce2_2d(&mean,
                   &variance,
                   (_FLOAT_ACCUM)1.0,
                   lcl_data,
                   xgrp_sz,
                   xlid,
                   ylid + zlid * ygrp_sz,
                   ygrp_sz * zgrp_sz);
#else
    local _FLOAT_ACCUM_C lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM_C lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)1.0, lcl_data_x, lcl_data_y, ylid + zlid * ygrp_sz);
#endif

    if(ylid == 0 && zlid == 0)
    {
        storeToStash(mean,
                     (__global _FLOAT_C*)meanvarbuff,
                     0,
                     zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                     ygrp_sz * ygrp_id * VEC_SIZE_Y,
                     ystride / VEC_SIZE_X,
                     xgrp_sz,
                     xgrp_id,
                     xlid,
                     xstride);
        storeToStash(variance,
                     (__global _FLOAT_C*)meanvarbuff,
                     1,
                     zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                     ygrp_sz * ygrp_id * VEC_SIZE_Y,
                     ystride / VEC_SIZE_X,
                     xgrp_sz,
                     xgrp_id,
                     xlid,
                     xstride);
    }
} // end spatial mean kernel

#endif // end USESAVED == 0

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormBwdSpatialDScaleDBias(const __global _FLOAT* __restrict x_in,
                                     const __global _FLOAT* __restrict dy_in,
                                     __global _FLOAT* __restrict buff,
                                     const __global _FLOAT_PREC* __restrict bnScale,
                                     const __global _FLOAT_PREC* __restrict bnBias,
#if MIO_BN_USESAVED == 1
                                     const __global _FLOAT_PREC* __restrict savedMean,
                                     const __global _FLOAT_PREC* __restrict savedInvVariance,
#endif
                                     _FLOAT_PREC _alpha,
                                     _FLOAT_PREC _beta)
{

    ACTIVATION_SET()

    unsigned int xlid    = get_local_id(0);
    unsigned int ylid    = get_local_id(1);
    unsigned int zlid    = get_local_id(2);
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int zgid    = get_global_id(2);
    unsigned int xgrp_id = get_group_id(0);
    unsigned int ygrp_id = get_group_id(1);
    unsigned int zgrp_id = get_group_id(2);
    unsigned int xgrp_sz = get_local_size(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int zgrp_sz = get_local_size(2);

    unsigned int xstride = MIO_LAYOUT_NHWC ? 1 : MIO_BN_HW;
    unsigned int ystride = MIO_LAYOUT_NHWC ? MIO_BN_C : 1;

    if(xgid * VEC_SIZE_X >= MIO_BN_C)
        return;

    unsigned int index;
    _FLOAT_PREC_C mean, invVar;
    _FLOAT_PREC_LS elemStd, xhat;
    _FLOAT_PREC_C dscale = (_FLOAT_PREC_C)0.;
    _FLOAT_PREC_C dbias  = (_FLOAT_PREC_C)0.;
    _FLOAT_PREC_C pscale = (_FLOAT_PREC_C)0.;
    _FLOAT_PREC_C pbias  = (_FLOAT_PREC_C)0.;

    local _FLOAT_PREC_C lmean[MIO_BN_GRP0], livar[MIO_BN_GRP0];
#if(MIOPEN_NRN_OP_ID > 0)
    local _FLOAT_PREC_C lcl_scale[MIO_BN_GRP0];
    local _FLOAT_PREC_C lcl_bias[MIO_BN_GRP0];
#endif

    if(ylid == 0 && zlid == 0)
    {
#if MIO_BN_USESAVED == 0
        lmean[xlid]     = loadFromStash((__global _FLOAT_C*)buff,
                                    0,
                                    zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                                    ygrp_sz * ygrp_id * VEC_SIZE_Y,
                                    ystride / VEC_SIZE_X,
                                    xgrp_sz,
                                    xgrp_id,
                                    xlid,
                                    xstride);
        livar[xlid]     = loadFromStash((__global _FLOAT_C*)buff,
                                    1,
                                    zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                                    ygrp_sz * ygrp_id * VEC_SIZE_Y,
                                    ystride / VEC_SIZE_X,
                                    xgrp_sz,
                                    xgrp_id,
                                    xlid,
                                    xstride);
#else
        lmean[xlid] = *((__global _FLOAT_PREC_C*)(savedMean + xgid * VEC_SIZE_X));
        livar[xlid] = *((__global _FLOAT_PREC_C*)(savedInvVariance + xgid * VEC_SIZE_X));
#endif
#if(MIOPEN_NRN_OP_ID > 0)
        lcl_scale[xlid] = *((__global _FLOAT_PREC_C*)(bnScale + xgid * VEC_SIZE_X));
        lcl_bias[xlid]  = *((__global _FLOAT_PREC_C*)(bnBias + xgid * VEC_SIZE_X));
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid * VEC_SIZE_Y < MIO_BN_HW && zgid < MIO_BN_N)
    {
        mean   = lmean[xlid];
        invVar = livar[xlid];
#if(MIOPEN_NRN_OP_ID > 0)
        pscale = lcl_scale[xlid];
        pbias  = lcl_bias[xlid];
#endif

        _FLOAT_LS read4;
        _FLOAT_PREC_LS value1, value2;
        unsigned int index_base = (zgid * MIO_BN_N_ELEMENTS) * MIO_BN_CHW +
                                  ygid * ystride * VEC_SIZE_Y + xgid * xstride * VEC_SIZE_X;
        for(unsigned int n = 0; n < MIO_BN_N_ELEMENTS; n++)
        {
            index   = index_base + n * MIO_BN_CHW;
            read4   = *((const __global _FLOAT_LS*)(dy_in + index));
            value1  = FLOAT2FLOATPREC_VEC(read4);
            read4   = *((const __global _FLOAT_LS*)(x_in + index));
            value2  = FLOAT2FLOATPREC_VEC(read4);
            elemStd = value2 - mean;
            xhat    = elemStd * invVar;
            // apply activation function on dy
            ACTIVATION_OP_BWD(value1, xhat, pscale, pbias, value1, _FLOAT_PREC_LS)
            _ACCUMULATE(dbias, value1)
            _ACCUMULATE_MAD(dscale, xhat, value1, dscale)
        }
    }

#if !MIOPEN_USE_AMDGCN || MIO_BN_GRP0 > 1 || MIO_BN_LDSGCN_SIZE == 1 || VEC_SIZE_X > 1
    local _FLOAT_ACCUM_C lcl_data[2 * MIO_BN_LDS_SIZE];
    lds_reduce2_2d(&dscale,
                   &dbias,
                   (_FLOAT_ACCUM)1.0,
                   lcl_data,
                   xgrp_sz,
                   xlid,
                   ylid + zlid * ygrp_sz,
                   ygrp_sz * zgrp_sz);
#else
    local _FLOAT_ACCUM_C lcl_data_x[MIO_BN_LDSGCN_SIZE];
    local _FLOAT_ACCUM_C lcl_data_y[MIO_BN_LDSGCN_SIZE];
    gcn_reduce2(&dscale, &dbias, (_FLOAT_ACCUM)1.0, lcl_data_x, lcl_data_y, ylid + zlid * ygrp_sz);
#endif

    if(ylid == 0 && zlid == 0)
    {
        const unsigned int stash_index = MIO_BN_USESAVED == 1 ? 0 : 2;
        storeToStash(dscale,
                     (__global _FLOAT_C*)buff,
                     stash_index,
                     zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                     ygrp_sz * ygrp_id * VEC_SIZE_Y,
                     ystride / VEC_SIZE_X,
                     xgrp_sz,
                     xgrp_id,
                     xlid,
                     xstride);
        storeToStash(dbias,
                     (__global _FLOAT_C*)buff,
                     stash_index + 1,
                     zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                     ygrp_sz * ygrp_id * VEC_SIZE_Y,
                     ystride / VEC_SIZE_X,
                     xgrp_sz,
                     xgrp_id,
                     xlid,
                     xstride);
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0_FINAL, MIO_BN_GRP1_FINAL, MIO_BN_GRP2_FINAL)))
__kernel void
MIOpenBatchNormBwdSpatialFinalDScaleDBias(const __global _FLOAT* __restrict buff,
                                          __global _FLOAT_PREC* __restrict delta_scale,
                                          __global _FLOAT_PREC* __restrict delta_bias)
{

    unsigned int xlid    = get_local_id(0);
    unsigned int ylid    = get_local_id(1);
    unsigned int zlid    = get_local_id(2);
    unsigned int xgid    = get_global_id(0);
    unsigned int xgrp_id = get_group_id(0);
    unsigned int xgrp_sz = get_local_size(0);
    unsigned int ygrp_sz = get_local_size(1);
    unsigned int zgrp_sz = get_local_size(2);

    unsigned int xstride           = MIO_LAYOUT_NHWC ? 1 : MIO_BN_HW;
    unsigned int ystride           = MIO_LAYOUT_NHWC ? MIO_BN_C : 1;
    const unsigned int stash_index = MIO_BN_USESAVED == 1 ? 0 : 2;

    if(xgid * VEC_SIZE_X >= MIO_BN_C)
        return;

    _FLOAT_PREC_C dscale = (_FLOAT_PREC_C)0.;
    _FLOAT_PREC_C dbias  = (_FLOAT_PREC_C)0.;

    for(unsigned int zoffset = zlid; zoffset < MIO_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < MIO_BN_NGRPS; yoffset += ygrp_sz)
        {
            dscale += loadFromStash((__global _FLOAT_C*)buff,
                                    stash_index,
                                    MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                                    MIO_BN_GRP1 * yoffset * VEC_SIZE_Y,
                                    ystride / VEC_SIZE_X,
                                    xgrp_sz,
                                    xgrp_id,
                                    xlid,
                                    xstride);
            dbias += loadFromStash((__global _FLOAT_C*)buff,
                                   stash_index + 1,
                                   MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                                   MIO_BN_GRP1 * yoffset * VEC_SIZE_Y,
                                   ystride / VEC_SIZE_X,
                                   xgrp_sz,
                                   xgrp_id,
                                   xlid,
                                   xstride);
        }
    }

#if !MIOPEN_USE_AMDGCN || MIO_BN_GRP0 > 1 || MIO_BN_LDSGCN_SIZE == 1 || VEC_SIZE_X > 1
    local _FLOAT_ACCUM_C lcl_data[2 * MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL];
    lds_reduce2_2d(&dscale,
                   &dbias,
                   (_FLOAT_ACCUM)1.0,
                   lcl_data,
                   xgrp_sz,
                   xlid,
                   ylid + zlid * ygrp_sz,
                   ygrp_sz * zgrp_sz);
#else
    local _FLOAT_ACCUM_C lcl_data_x[MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL / 64];
    local _FLOAT_ACCUM_C lcl_data_y[MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL / 64];
    gcn_reduce2(&dscale, &dbias, (_FLOAT_ACCUM)1.0, lcl_data_x, lcl_data_y, ylid + zlid * ygrp_sz);
#endif

    if(ylid == 0 && zlid == 0)
    {
        *((__global _FLOAT_PREC_C*)(delta_scale + xgid * VEC_SIZE_X)) = dscale;
        *((__global _FLOAT_PREC_C*)(delta_bias + xgid * VEC_SIZE_X))  = dbias;
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormBwdSpatialDX(const __global _FLOAT* __restrict x_in,
                            const __global _FLOAT* __restrict dy_in,
                            __global _FLOAT* __restrict dx_out,
                            const __global _FLOAT_PREC* __restrict bnScale,
                            const __global _FLOAT_PREC* __restrict bnBias,
                            const __global _FLOAT_PREC* __restrict delta_scale,
                            const __global _FLOAT_PREC* __restrict delta_bias,
#if MIO_BN_USESAVED == 1
                            const __global _FLOAT_PREC* __restrict savedMean,
                            const __global _FLOAT_PREC* __restrict savedInvVariance,
#endif
                            _FLOAT_PREC INHW,
                            _FLOAT_PREC _alpha,
                            _FLOAT_PREC _beta)
{

    ACTIVATION_SET()
    unsigned int xlid = get_local_id(0);
    unsigned int ylid = get_local_id(1);
    unsigned int zlid = get_local_id(2);
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);
    unsigned int zgid = get_global_id(2);

    unsigned int xstride = MIO_LAYOUT_NHWC ? 1 : MIO_BN_HW;
    unsigned int ystride = MIO_LAYOUT_NHWC ? MIO_BN_C : 1;

    if(xgid * VEC_SIZE_X >= MIO_BN_C)
        return;

    unsigned int index;
    _FLOAT_PREC_C mean, invVar;
    _FLOAT_PREC_LS elemStd, xhat;
    _FLOAT_PREC_C pscale, dscale, dbias;
    _FLOAT_PREC_C pbias = (_FLOAT_PREC_C)0.;
    _FLOAT_PREC_LS tmp1, tmp2, tmp3, tmp4;
    _FLOAT_PREC_LS value1;
    _FLOAT_LS read4;
    _FLOAT_PREC NHW = (_FLOAT_PREC)MIO_BN_NHW;

    local _FLOAT_PREC_C lscale[MIO_BN_GRP0], ldscale[MIO_BN_GRP0], ldbias[MIO_BN_GRP0],
        lmean[MIO_BN_GRP0], livar[MIO_BN_GRP0];
#if(MIOPEN_NRN_OP_ID > 0)
    local _FLOAT_PREC_C lbias[MIO_BN_GRP0];
#endif

    if(ylid == 0 && zlid == 0)
    {
#if MIO_BN_USESAVED == 0
        unsigned int xgrp_id = get_group_id(0);
        unsigned int ygrp_id = get_group_id(1);
        unsigned int zgrp_id = get_group_id(2);
        unsigned int xgrp_sz = get_local_size(0);
        unsigned int ygrp_sz = get_local_size(1);
        unsigned int zgrp_sz = get_local_size(2);

        lmean[xlid]   = loadFromStash((__global _FLOAT_C*)dx_out,
                                    0,
                                    zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                                    ygrp_sz * ygrp_id * VEC_SIZE_Y,
                                    ystride / VEC_SIZE_X,
                                    xgrp_sz,
                                    xgrp_id,
                                    xlid,
                                    xstride);
        livar[xlid]   = loadFromStash((__global _FLOAT_C*)dx_out,
                                    1,
                                    zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                                    ygrp_sz * ygrp_id * VEC_SIZE_Y,
                                    ystride / VEC_SIZE_X,
                                    xgrp_sz,
                                    xgrp_id,
                                    xlid,
                                    xstride);
#else
        lmean[xlid] = *((const __global _FLOAT_PREC_C*)(savedMean + xgid * VEC_SIZE_X));
        livar[xlid] = *((const __global _FLOAT_PREC_C*)(savedInvVariance + xgid * VEC_SIZE_X));
#endif
        lscale[xlid]  = *((const __global _FLOAT_PREC_C*)(bnScale + xgid * VEC_SIZE_X));
#if(MIOPEN_NRN_OP_ID > 0)
        lbias[xlid]   = *((const __global _FLOAT_PREC_C*)(bnBias + xgid * VEC_SIZE_X));
#endif
        ldscale[xlid] = *((const __global _FLOAT_PREC_C*)(delta_scale + xgid * VEC_SIZE_X));
        ldbias[xlid]  = *((const __global _FLOAT_PREC_C*)(delta_bias + xgid * VEC_SIZE_X));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(ygid * VEC_SIZE_Y < MIO_BN_HW && zgid < MIO_BN_N)
    {
        mean   = lmean[xlid];
        invVar = livar[xlid];
        pscale = lscale[xlid];
#if(MIOPEN_NRN_OP_ID > 0)
        pbias  = lbias[xlid];
#endif
        dscale = ldscale[xlid];
        dbias  = ldbias[xlid];

        unsigned int index_base = (zgid * MIO_BN_N_ELEMENTS) * MIO_BN_CHW +
                                  ygid * ystride * VEC_SIZE_Y + xgid * xstride * VEC_SIZE_X;
        for(unsigned int n = 0; n < MIO_BN_N_ELEMENTS; n++)
        { // apply normalization
            index   = index_base + n * MIO_BN_CHW;
            read4   = *((const __global _FLOAT_LS*)(x_in + index));
            value1  = FLOAT2FLOATPREC_VEC(read4);
            elemStd = value1 - mean;    // (x_i - mean)
            xhat    = elemStd * invVar; // recalculating this again...
            read4   = *((const __global _FLOAT_LS*)(dy_in + index));
            value1  = FLOAT2FLOATPREC_VEC(read4);
            ACTIVATION_OP_BWD(value1, xhat, pscale, pbias, value1, _FLOAT_PREC_LS)
            tmp1                                     = mad((_FLOAT_PREC_LS)NHW, value1, -dbias);
            tmp2                                     = -xhat * dscale;
            tmp3                                     = pscale * invVar * INHW;
            tmp4                                     = tmp3 * (tmp2 + tmp1);
            *((__global _FLOAT_LS*)(dx_out + index)) = FLOATPREC2FLOAT_VEC(tmp4);
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
                          const __global _FLOAT_PREC* __restrict bnScale,
                          const __global _FLOAT_PREC* __restrict bnBias,
                          __global _FLOAT_PREC* __restrict dscale,
                          __global _FLOAT_PREC* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                          double epsilon,
#elif(MIO_BN_USESAVED == 1)
                          const __global _FLOAT_PREC* savedMean,
                          const __global _FLOAT_PREC* savedInvVariance,
#endif
                          _FLOAT_PREC INHW,
                          _FLOAT_PREC _alpha,
                          _FLOAT_PREC _beta)
{

    ACTIVATION_SET()
    // SPATIAL
    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
#if(MIO_BN_USESAVED == 0)
    _FLOAT_PREC variance    = (_FLOAT_PREC)0.;
#endif
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pscale      = (_FLOAT_PREC)0.;
    _FLOAT_PREC pbias       = (_FLOAT_PREC)0.;
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
#if(MIOPEN_NRN_OP_ID > 0)
    __local _FLOAT_PREC lcl_bias;
#endif

    if(lid == 0)
    {
        lcl_scale = *(bnScale + grpid);
#if(MIOPEN_NRN_OP_ID > 0)
        lcl_bias  = *(bnBias + grpid);
#endif

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

    pscale = lcl_scale;
#if(MIOPEN_NRN_OP_ID > 0)
    pbias  = lcl_bias;
#endif

    if(lid < MIO_BN_HW)
    {
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index          = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            dyvalues[n]    = FLOAT2FLOATPREC(*(dy_in + index));

#if(MIO_BN_USESAVED == 1)
            batchvalues[n] = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
#else
            batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif // batchvalues is now xhat

            ACTIVATION_OP_BWD(dyvalues[n], batchvalues[n], pscale, pbias, dyvalues[n], _FLOAT_PREC)
            db += dyvalues[n];
            ds = mad(batchvalues[n], dyvalues[n], ds);
#else  // maxn
            _FLOAT_PREC dyvalue = FLOAT2FLOATPREC(*(dy_in + index));
            _FLOAT_PREC xhat    = ((FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance);
            ACTIVATION_OP_BWD(dyvalue, xhat, pscale, pbias, dyvalue, _FLOAT_PREC)
            db += dyvalue;
            ds = mad(xhat, dyvalue, ds);
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
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index         = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            tmp1          = mad(NHW, dyvalues[n], -db);
            tmp2          = -(batchvalues[n]) * ds;
#else
            _FLOAT_PREC dyvalue = FLOAT2FLOATPREC(*(dy_in + index));
            _FLOAT_PREC xhat    = ((FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance);
            ACTIVATION_OP_BWD(dyvalue, xhat, pscale, pbias, dyvalue, _FLOAT_PREC)

            tmp1 = mad(NHW, dyvalue, -db);
            tmp2 = -xhat * ds;
#endif
            tmp3          = (pscale * invVariance) * INHW;
            tmp3          = tmp3 * (tmp2 + tmp1);
            dx_out[index] = FLOATPREC2FLOAT(tmp3);
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
