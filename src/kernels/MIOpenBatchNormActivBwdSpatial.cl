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
MIOpenBatchNormActivBwdSpatial(const __global _FLOAT* __restrict x_in,
                               const __global _FLOAT* __restrict y_in,
                               const __global _FLOAT* __restrict dy_in,
                               __global _FLOAT* __restrict dx_out,
                               _FLOAT diff_scale,
                               _FLOAT gamma,
                               _FLOAT beta,
                               _FLOAT alpha,
                               const __global _FLOAT_PREC* __restrict bnScale,
                               const __global _FLOAT_PREC* __restrict bnBias,
                               __global _FLOAT_PREC* __restrict dscale,
                               __global _FLOAT_PREC* __restrict dbias,
                               const __global _FLOAT_PREC* __restrict savedMean,
                               const __global _FLOAT_PREC* __restrict savedInvVariance,
                               float INHW
#if MIO_BN_CBA_WRITE_INTERMEDIATE
                               ,
                               __global _FLOAT* __restrict bn_out_dev,
                               __global _FLOAT* __restrict bn_dyin_dev
#endif
)
{

    // SPATIAL
    _FLOAT_PREC mean = (_FLOAT_PREC)0.;

    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pscale      = (_FLOAT_PREC)0.;
    _FLOAT_PREC ds          = (_FLOAT_PREC)0.;
    _FLOAT_PREC db          = (_FLOAT_PREC)0.;

    _FLOAT_PREC batchvalues[MIO_BN_NLOOP];
    _FLOAT_PREC dyvalues[MIO_BN_NLOOP];
    __local _FLOAT_PREC lbns, lbnb;
    __local _FLOAT_PREC lmean, lvar;

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
        lbns  = *(bnScale + grpid);
        lbnb  = *(bnBias + grpid);
        lmean = *(savedMean + grpid);
        lvar  = *(savedInvVariance + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean        = lmean;
    invVariance = lvar;
    //-------------------------------------------

    //==== CALC DB and DS =========================================
    if(lid < MIO_BN_SEGMENT)
    {

        for(unsigned int n = 0; n < MIO_BN_NLOOPM; ++n)
        {
            nid                = n * MIO_BN_SEGIHW + lidihw;
            index              = nid * MIO_BN_CHW + chwid;
            _FLOAT_PREC xhat   = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
            _FLOAT_PREC bn_out = mad(xhat, lbns, lbnb);
            _FLOAT_PREC bn_dyin;
            _FLOAT_PREC act_dyin = FLOAT2FLOATPREC(*(dy_in + index));
            _FLOAT_PREC act_out  = FLOAT2FLOATPREC(*(y_in + index));
            ActivationFunction_Diff(1,
                                    &bn_dyin,
                                    &act_dyin,
                                    &bn_out,
                                    &act_out,
                                    FLOAT2FLOATPREC(diff_scale),
                                    FLOAT2FLOATPREC(gamma),
                                    FLOAT2FLOATPREC(beta),
                                    FLOAT2FLOATPREC(alpha));
            dyvalues[n] = bn_dyin;
            db += dyvalues[n];
            batchvalues[n] = xhat;
#if MIO_BN_CBA_WRITE_INTERMEDIATE
            // for debugging
            bn_out_dev[index]  = bn_out;
            bn_dyin_dev[index] = bn_dyin;
#endif
            // batchvalues is now xhat
            ds = mad(batchvalues[n], dyvalues[n], ds);
        }
        nid   = MIO_BN_SNHW + lidihw;
        index = nid * MIO_BN_CHW + chwid;
        if(index < MIO_BN_NCHW)
        {
            _FLOAT_PREC xhat   = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
            _FLOAT_PREC bn_out = mad(xhat, lbns, lbnb);
            _FLOAT_PREC bn_dyin;
            _FLOAT_PREC act_dyin = FLOAT2FLOATPREC(*(dy_in + index));
            _FLOAT_PREC act_out  = FLOAT2FLOATPREC(*(y_in + index));
            ActivationFunction_Diff(1,
                                    &bn_dyin,
                                    &act_dyin,
                                    &bn_out,
                                    &act_out,
                                    FLOAT2FLOATPREC(diff_scale),
                                    FLOAT2FLOATPREC(gamma),
                                    FLOAT2FLOATPREC(beta),
                                    FLOAT2FLOATPREC(alpha));
            dyvalues[MIO_BN_NLOOPM] = bn_dyin;

#if MIO_BN_CBA_WRITE_INTERMEDIATE
            // for debugging
            bn_out_dev[index]  = bn_out;
            bn_dyin_dev[index] = bn_dyin;
#endif
        }
        else
        {
            dyvalues[MIO_BN_NLOOPM] = (_FLOAT_PREC)0.;
        }
        db += dyvalues[MIO_BN_NLOOPM];

        batchvalues[MIO_BN_NLOOPM] = (index < MIO_BN_NCHW)
                                         ? (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance
                                         : (_FLOAT_PREC)0.;

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
MIOpenBatchNormActivBwdSpatial(const __global _FLOAT* __restrict x_in,
                               const __global _FLOAT* __restrict y_in,
                               const __global _FLOAT* __restrict dy_in,
                               __global _FLOAT* __restrict dx_out,
                               _FLOAT diff_scale,
                               _FLOAT gamma,
                               _FLOAT beta,
                               _FLOAT alpha,
                               const __global _FLOAT_PREC* __restrict bnScale,
                               const __global _FLOAT_PREC* __restrict bnBias,
                               __global _FLOAT_PREC* __restrict dscale,
                               __global _FLOAT_PREC* __restrict dbias,
                               const __global _FLOAT_PREC* __restrict savedMean,
                               const __global _FLOAT_PREC* __restrict savedInvVariance,
                               float INHW
#if MIO_BN_CBA_WRITE_INTERMEDIATE
                               ,
                               __global _FLOAT* __restrict bn_out_dev,
                               __global _FLOAT* __restrict bn_dyin_dev
#endif
)
{

    // SPATIAL
    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC pscale      = (_FLOAT_PREC)0.;
    _FLOAT_PREC db          = (_FLOAT_PREC)0.;
    _FLOAT_PREC ds          = (_FLOAT_PREC)0.;
    _FLOAT_PREC xhat        = (_FLOAT_PREC)0.;

    __local _FLOAT_PREC lmean, lvar;
    __local _FLOAT_PREC lcl_scale, lcl_bias;
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
        lcl_bias  = *(bnBias + grpid);
        lmean     = *(savedMean + grpid);
        lvar      = *(savedInvVariance + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean        = lmean;
    invVariance = lvar;

    _FLOAT4 xread4;
    _FLOAT_PREC4 xhat4;
    _FLOAT4 act_dyin4, act_out4;
    _FLOAT_PREC4 bn_out4;

    for(unsigned int k = lid << 2; k < MIO_BN_LESS4; k += GRPRD)
    {
        nidx      = k / MIO_BN_HW;
        hwidx     = k - (nidx * MIO_BN_HW);
        index     = nidx * MIO_BN_CHW + chwid + hwidx;
        xread4    = *((const global _FLOAT4*)(x_in + index));
        act_dyin4 = *((const global _FLOAT4*)(dy_in + index));
        act_out4  = *((const global _FLOAT4*)(y_in + index));
        xhat4.x   = (FLOAT2FLOATPREC(xread4.x) - mean) * invVariance;
        xhat4.y   = (FLOAT2FLOATPREC(xread4.y) - mean) * invVariance;
        xhat4.z   = (FLOAT2FLOATPREC(xread4.z) - mean) * invVariance;
        xhat4.w   = (FLOAT2FLOATPREC(xread4.w) - mean) * invVariance;

        bn_out4.x = mad(xhat4.x, lcl_scale, lcl_bias);
        bn_out4.y = mad(xhat4.y, lcl_scale, lcl_bias);
        bn_out4.z = mad(xhat4.z, lcl_scale, lcl_bias);
        bn_out4.w = mad(xhat4.w, lcl_scale, lcl_bias);

        _FLOAT_PREC pbndyin  = 0.;
        _FLOAT_PREC pactdyin = act_dyin4.x;
        _FLOAT_PREC pbnout   = bn_out4.x;
        _FLOAT_PREC pactout  = act_out4.x;
        ActivationFunction_Diff(1,
                                &pbndyin,
                                &pactdyin,
                                &pbnout,
                                &pactout,
                                FLOAT2FLOATPREC(diff_scale),
                                FLOAT2FLOATPREC(gamma),
                                FLOAT2FLOATPREC(beta),
                                FLOAT2FLOATPREC(alpha));

        db += pbndyin;
        ds       = mad(xhat4.x, pbndyin, ds);
        pactdyin = act_dyin4.y;
        pbnout   = bn_out4.y;
        pactout  = act_out4.y;
        ActivationFunction_Diff(1,
                                &pbndyin,
                                &pactdyin,
                                &pbnout,
                                &pactout,
                                FLOAT2FLOATPREC(diff_scale),
                                FLOAT2FLOATPREC(gamma),
                                FLOAT2FLOATPREC(beta),
                                FLOAT2FLOATPREC(alpha));

        db += pbndyin;
        ds       = mad(xhat4.y, pbndyin, ds);
        pactdyin = act_dyin4.z;
        pbnout   = bn_out4.z;
        pactout  = act_out4.z;
        ActivationFunction_Diff(1,
                                &pbndyin,
                                &pactdyin,
                                &pbnout,
                                &pactout,
                                FLOAT2FLOATPREC(diff_scale),
                                FLOAT2FLOATPREC(gamma),
                                FLOAT2FLOATPREC(beta),
                                FLOAT2FLOATPREC(alpha));
        db += pbndyin;
        ds       = mad(xhat4.z, pbndyin, ds);
        pactdyin = act_dyin4.w;
        pbnout   = bn_out4.w;
        pactout  = act_out4.w;
        ActivationFunction_Diff(1,
                                &pbndyin,
                                &pactdyin,
                                &pbnout,
                                &pactout,
                                FLOAT2FLOATPREC(diff_scale),
                                FLOAT2FLOATPREC(gamma),
                                FLOAT2FLOATPREC(beta),
                                FLOAT2FLOATPREC(alpha));
        db += pbndyin;
        ds = mad(xhat4.w, pbndyin, ds);

#if MIO_BN_CBA_WRITE_INTERMEDIATE
        // for debugging
        bn_out_dev[index]     = bn_out4.x;
        bn_out_dev[index + 1] = bn_out4.y;
        bn_out_dev[index + 2] = bn_out4.z;
        bn_out_dev[index + 3] = bn_out4.w;

        bn_dyin_dev[index]     = bn_dyin4.x;
        bn_dyin_dev[index + 1] = bn_dyin4.y;
        bn_dyin_dev[index + 2] = bn_dyin4.z;
        bn_dyin_dev[index + 3] = bn_dyin4.w;
#endif
    }

#if(MIO_BN_REM4)
    unsigned int remkey = (lid << 2) + MIO_BN_LESS4;
    nidx                = remkey / MIO_BN_HW;
    hwidx               = remkey - (nidx * MIO_BN_HW);
    index               = nidx * MIO_BN_CHW + chwid + hwidx;
    if(index < MIO_BN_NCHW)
    {
        xread4    = *((const global _FLOAT4*)(x_in + index));
        act_dyin4 = *((const global _FLOAT4*)(dy_in + index));
        act_out4  = *((const global _FLOAT4*)(y_in + index));
        xhat4.x   = (FLOAT2FLOATPREC(xread4.x) - mean) * invVariance;
        xhat4.y   = (FLOAT2FLOATPREC(xread4.y) - mean) * invVariance;
        xhat4.z   = (FLOAT2FLOATPREC(xread4.z) - mean) * invVariance;
        xhat4.w   = (FLOAT2FLOATPREC(xread4.w) - mean) * invVariance;

        bn_out4.x = mad(xhat4.x, lcl_scale, lcl_bias);
        bn_out4.y = mad(xhat4.y, lcl_scale, lcl_bias);
        bn_out4.z = mad(xhat4.z, lcl_scale, lcl_bias);
        bn_out4.w = mad(xhat4.w, lcl_scale, lcl_bias);

        _FLOAT_PREC pbndyin  = 0.;
        _FLOAT_PREC pactdyin = act_dyin4.x;
        _FLOAT_PREC pbnout   = bn_out4.x;
        _FLOAT_PREC pactout  = act_out4.x;
        ActivationFunction_Diff(1,
                                &pbndyin,
                                &pactdyin,
                                &pbnout,
                                &pactout,
                                FLOAT2FLOATPREC(diff_scale),
                                FLOAT2FLOATPREC(gamma),
                                FLOAT2FLOATPREC(beta),
                                FLOAT2FLOATPREC(alpha));

        db += pbndyin;
        ds       = mad(xhat4.x, pbndyin, ds);
        pactdyin = act_dyin4.y;
        pbnout   = bn_out4.y;
        pactout  = act_out4.y;
        ActivationFunction_Diff(1,
                                &pbndyin,
                                &pactdyin,
                                &pbnout,
                                &pactout,
                                FLOAT2FLOATPREC(diff_scale),
                                FLOAT2FLOATPREC(gamma),
                                FLOAT2FLOATPREC(beta),
                                FLOAT2FLOATPREC(alpha));

        db += pbndyin;
        ds       = mad(xhat4.y, pbndyin, ds);
        pactdyin = act_dyin4.z;
        pbnout   = bn_out4.z;
        pactout  = act_out4.z;
        ActivationFunction_Diff(1,
                                &pbndyin,
                                &pactdyin,
                                &pbnout,
                                &pactout,
                                FLOAT2FLOATPREC(diff_scale),
                                FLOAT2FLOATPREC(gamma),
                                FLOAT2FLOATPREC(beta),
                                FLOAT2FLOATPREC(alpha));
        db += pbndyin;
        ds       = mad(xhat4.z, pbndyin, ds);
        pactdyin = act_dyin4.w;
        pbnout   = bn_out4.w;
        pactout  = act_out4.w;
        ActivationFunction_Diff(1,
                                &pbndyin,
                                &pactdyin,
                                &pbnout,
                                &pactout,
                                FLOAT2FLOATPREC(diff_scale),
                                FLOAT2FLOATPREC(gamma),
                                FLOAT2FLOATPREC(beta),
                                FLOAT2FLOATPREC(alpha));
        db += pbndyin;
        ds = mad(xhat4.w, pbndyin, ds);

#if MIO_BN_CBA_WRITE_INTERMEDIATE
        // for debugging
        bn_out_dev[index]     = bn_out4.x;
        bn_out_dev[index + 1] = bn_out4.y;
        bn_out_dev[index + 2] = bn_out4.z;
        bn_out_dev[index + 3] = bn_out4.w;

        bn_dyin_dev[index]     = bn_dyin4.x;
        bn_dyin_dev[index + 1] = bn_dyin4.y;
        bn_dyin_dev[index + 2] = bn_dyin4.z;
        bn_dyin_dev[index + 3] = bn_dyin4.w;
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
        *(dbias + grpid)  = (_FLOAT_PREC)db;
        *(dscale + grpid) = (_FLOAT_PREC)ds;
    }

    _FLOAT_PREC vals[MIO_MAX_READ];
    for(unsigned int k = (MIO_MAX_READ * lid); k < MIO_BN_LESSOUT; k += MIO_BN_CHUNK)
    {
        for(unsigned int j = 0; j < MIO_MAX_READ; j++)
        {
            unsigned int l = k + j;
            nidx           = l / MIO_BN_HW;
            hwidx          = l - (nidx * MIO_BN_HW);
            index          = nidx * MIO_BN_CHW + chwid + hwidx;
            _FLOAT_PREC bn_dyin;
            _FLOAT_PREC act_dyin = FLOAT2FLOATPREC(*(dy_in + index));
            _FLOAT_PREC act_out  = FLOAT2FLOATPREC(*(y_in + index));
            xhat                 = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
            _FLOAT_PREC bn_out   = mad(xhat, lcl_scale, lcl_bias);
            ActivationFunction_Diff(1,
                                    &bn_dyin,
                                    &act_dyin,
                                    &bn_out,
                                    &act_out,
                                    FLOAT2FLOATPREC(diff_scale),
                                    FLOAT2FLOATPREC(gamma),
                                    FLOAT2FLOATPREC(beta),
                                    FLOAT2FLOATPREC(alpha));
            tmp1    = mad(NHW, bn_dyin, -db);
            tmp2    = -xhat * ds;
            vals[j] = tmp3 * (tmp2 + tmp1);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for(unsigned int j = 0; j < MIO_MAX_READ; j++)
        {
            unsigned int l    = k + j;
            nidx              = l / MIO_BN_HW;
            hwidx             = l - (nidx * MIO_BN_HW);
            index             = nidx * MIO_BN_CHW + chwid + hwidx;
            *(dx_out + index) = FLOATPREC2FLOAT(vals[j]);
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
            _FLOAT_PREC bn_dyin;
            _FLOAT_PREC act_dyin = FLOAT2FLOATPREC(*(dy_in + index));
            _FLOAT_PREC act_out  = FLOAT2FLOATPREC(*(y_in + index));
            xhat                 = (*(x_in + index) - mean) * invVariance;
            _FLOAT_PREC bn_out   = mad(xhat, lcl_scale, lcl_bias);
            ActivationFunction_Diff(1,
                                    &bn_dyin,
                                    &act_dyin,
                                    &bn_out,
                                    &act_out,
                                    FLOAT2FLOATPREC(diff_scale),
                                    FLOAT2FLOATPREC(gamma),
                                    FLOAT2FLOATPREC(beta),
                                    FLOAT2FLOATPREC(alpha));

            tmp1    = mad(NHW, bn_dyin, -db);
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
            *(dx_out + index) = FLOATPREC2FLOAT(vals[j]);
        }
    }
#endif
}

#elif(MIO_BN_VARIANT == 2)
//============================================================

#elif(MIO_BN_VARIANT == 3)

//=============== SINGLE WORKGROUP PER CHANNEL ===============//
__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormActivBwdSpatial(const __global _FLOAT* __restrict x_in,
                               const __global _FLOAT* __restrict y_in,
                               const __global _FLOAT* __restrict dy_in,
                               __global _FLOAT* __restrict dx_out,
                               _FLOAT diff_scale,
                               _FLOAT gamma,
                               _FLOAT beta,
                               _FLOAT alpha,
                               const __global _FLOAT_PREC* __restrict bnScale,
                               const __global _FLOAT_PREC* __restrict bnBias,
                               __global _FLOAT_PREC* __restrict dscale,
                               __global _FLOAT_PREC* __restrict dbias,
                               const __global _FLOAT_PREC* __restrict savedMean,
                               const __global _FLOAT_PREC* __restrict savedInvVariance,
                               float INHW
#if MIO_BN_CBA_WRITE_INTERMEDIATE
                               ,
                               __global _FLOAT* __restrict bn_out_dev,
                               __global _FLOAT* __restrict bn_dyin_dev
#endif
)
{
    unsigned int lid        = get_local_id(0);
    unsigned int grpid      = get_group_id(0);
    unsigned int index      = 0;
    unsigned int cidx       = grpid * MIO_BN_HW;
    _FLOAT_PREC tmp1        = (_FLOAT_PREC)0.;
    _FLOAT_PREC tmp2        = (_FLOAT_PREC)0.;
    _FLOAT_PREC tmp3        = (_FLOAT_PREC)0.;
    _FLOAT_PREC ds          = (_FLOAT_PREC)0.;
    _FLOAT_PREC db          = (_FLOAT_PREC)0.;
    _FLOAT_PREC pscale      = (_FLOAT_PREC)0.;
    _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
    _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
    _FLOAT_PREC NHW         = (_FLOAT_PREC)MIO_BN_NHW;

    local _FLOAT_PREC lmean, lvar;
    local _FLOAT_PREC lcl_scale, lcl_bias;

    if(lid == 0)
    {
        lcl_scale = *(bnScale + grpid);
        lcl_bias  = *(bnBias + grpid);
        lmean     = *(savedMean + grpid);
        lvar      = *(savedInvVariance + grpid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mean        = lmean;
    invVariance = lvar;

#if(MIO_BN_N < MIO_BN_MAXN)
    _FLOAT_PREC batchvalues[MIO_BN_N];
    _FLOAT_PREC dyvalues[MIO_BN_N];
#endif

    if(lid < MIO_BN_HW)
    {
#pragma unroll
        for(unsigned n = 0; n < MIO_BN_N; n++)
        {
            index              = n * MIO_BN_CHW + cidx + lid;
            _FLOAT_PREC xhat   = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
            _FLOAT_PREC bn_out = mad(xhat, lcl_scale, lcl_bias);
            _FLOAT_PREC bn_dyin;
            _FLOAT_PREC act_dyin = FLOAT2FLOATPREC(*(dy_in + index));
            _FLOAT_PREC act_out  = FLOAT2FLOATPREC(*(y_in + index));
            ActivationFunction_Diff(1,
                                    &bn_dyin,
                                    &act_dyin,
                                    &bn_out,
                                    &act_out,
                                    FLOAT2FLOATPREC(diff_scale),
                                    FLOAT2FLOATPREC(gamma),
                                    FLOAT2FLOATPREC(beta),
                                    FLOAT2FLOATPREC(alpha));

#if MIO_BN_CBA_WRITE_INTERMEDIATE
            // for debugging
            bn_out_dev[index]  = bn_out;
            bn_dyin_dev[index] = bn_dyin;
#endif

#if(MIO_BN_N < MIO_BN_MAXN)
            batchvalues[n]     = xhat;
            dyvalues[n]        = (_FLOAT_PREC)bn_dyin;
#endif
            db += bn_dyin;
            ds = mad(xhat, bn_dyin, ds);
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
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    // Group level reduction
    // Need to reduce over all elements in NxHxW
    // move across the sections of an image in the mini_batch stack
    if(lid < MIO_BN_HW)
    {
        pscale = lcl_scale;

#pragma unroll
        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            index         = n * MIO_BN_CHW + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
            tmp1          = mad(NHW, dyvalues[n], -db);
            tmp2          = -(batchvalues[n] * ds);
#else
            _FLOAT_PREC act_dyin = FLOAT2FLOATPREC(*(dy_in + index));
            _FLOAT_PREC act_out  = FLOAT2FLOATPREC(*(y_in + index));
            _FLOAT_PREC xhat     = (FLOAT2FLOATPREC(*(x_in + index)) - mean) * invVariance;
            _FLOAT_PREC bn_out   = mad(xhat, lcl_scale, lcl_bias);
            _FLOAT_PREC bn_dyin;
            ActivationFunction_Diff(1,
                                    &bn_dyin,
                                    &act_dyin,
                                    &bn_out,
                                    &act_out,
                                    FLOAT2FLOATPREC(diff_scale),
                                    FLOAT2FLOATPREC(gamma),
                                    FLOAT2FLOATPREC(beta),
                                    FLOAT2FLOATPREC(alpha));

            tmp1 = mad(NHW, bn_dyin, -db);
            tmp2 = -(xhat)*ds;
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
