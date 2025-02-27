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

#include "batchnorm_functions.h"

__kernel void MIOpenBatchNormBwdPerActivationSaved(const __global _FLOAT* in,
                                                   const __global _FLOAT* dy_in,
                                                   unsigned int N,
                                                   unsigned int in_nstride,
                                                   unsigned int in_cstride,
                                                   __global _FLOAT* dx_out,
                                                   const __global _FLOAT_PREC* scale,
                                                   __global _FLOAT_PREC* delta_scale,
                                                   __global _FLOAT_PREC* delta_bias,
                                                   const __global _FLOAT_PREC* savedMean,
                                                   const __global _FLOAT_PREC* savedInvVariance)
{

    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int yglb_sz = get_global_size(1);
    int cidx    = in_cstride * xgid;

    unsigned int index, adjIndex;
    _FLOAT_PREC mean, invVar;
    _FLOAT_PREC xhat, dyelem;
    _FLOAT_PREC pvt_scale, pvt_dscale;
    _FLOAT_PREC pvt_dbias;
    _FLOAT_PREC tmp1, tmp2, tmp3;
    _FLOAT_PREC dxhat    = (_FLOAT_PREC)0.;
    _FLOAT_PREC dxhathat = (_FLOAT_PREC)0.;

    // move across the sections of an image in the mini_batch stack
    for(int idx = ygid; idx < in_cstride; idx += yglb_sz)
    {
        adjIndex   = cidx + idx;
        mean       = savedMean[adjIndex];
        invVar     = savedInvVariance[adjIndex];
        pvt_scale  = scale[adjIndex];
        pvt_dscale = (_FLOAT_PREC)0.;
        pvt_dbias  = (_FLOAT_PREC)0.;
        dxhat      = (_FLOAT_PREC)0.;
        dxhathat   = (_FLOAT_PREC)0.;

        for(int n = 0; n < N; n++)
        {
            // per (x-dims) channel load a block of data into LDS
            index  = in_nstride * n + adjIndex;
            xhat   = (FLOAT2FLOATPREC(in[index]) - mean) * invVar;
            dyelem = FLOAT2FLOATPREC(dy_in[index]);
            pvt_dbias += dyelem;
            pvt_dscale = mad(xhat, dyelem, pvt_dscale);
            tmp1       = pvt_scale * dyelem;
            dxhat += tmp1;
            dxhathat = mad(tmp1, xhat, dxhathat);
        } // end for(n)

        for(int n = 0; n < N; n++)
        {
            index         = in_nstride * n + adjIndex;
            xhat          = (FLOAT2FLOATPREC(in[index]) - mean) * invVar;
            tmp1          = mad(xhat, dxhathat, dxhat);
            tmp2          = mad((_FLOAT_PREC)N, FLOAT2FLOATPREC(dy_in[index]) * pvt_scale, -tmp1);
            tmp3          = invVar / ((_FLOAT_PREC)N);
            dx_out[index] = FLOATPREC2FLOAT(tmp3 * tmp2);
        }
        // Write out data
        delta_bias[adjIndex]  = pvt_dbias;
        delta_scale[adjIndex] = pvt_dscale;
    } // end for(img_offset) //image mini_batch is processed
}

__kernel void MIOpenBatchNormBwdPerActivation(const __global _FLOAT* in,
                                              const __global _FLOAT* dy_in,
                                              unsigned int N,
                                              unsigned int in_nstride,
                                              unsigned int in_cstride,
                                              __global _FLOAT* dx_out,
                                              const __global _FLOAT_PREC* scale,
                                              __global _FLOAT_PREC* delta_scale,
                                              __global _FLOAT_PREC* delta_bias,
                                              double epsilon)
{

    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int yglb_sz = get_global_size(1);
    int cidx    = in_cstride * xgid;

    unsigned int index, adjIndex;
    _FLOAT_PREC mean, invVar;
    _FLOAT_PREC xhat, dyelem;
    _FLOAT_PREC pvt_scale, pvt_dscale;
    _FLOAT_PREC pvt_dbias;
    _FLOAT_ACCUM tmp1, tmp2, tmp3;
    _FLOAT_PREC dxhat     = (_FLOAT_PREC)0.;
    _FLOAT_PREC dxhathat  = (_FLOAT_PREC)0.;
    _FLOAT_ACCUM variance = 0.;

    // move across the sections of the image mini_batch stack
    for(int idx = ygid; idx < in_cstride; idx += yglb_sz)
    {
        mean     = (_FLOAT_PREC)0.;
        adjIndex = cidx + idx; // gamma and beta tensor index
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = in_nstride * n + adjIndex;
            mean += FLOAT2FLOATPREC(in[index]);
        } // end for(n)
        mean /= (_FLOAT_PREC)N;
        variance = 0.;

        for(int n = 0; n < MIO_BN_N; n++)
        {
            index             = in_nstride * n + adjIndex;
            _FLOAT_PREC xdiff = FLOAT2FLOATPREC(in[index]) - mean;
            variance += (xdiff * xdiff);
        } // end for(n)
        variance /= (_FLOAT_PREC)N;
        invVar = rsqrt(variance + epsilon);

        pvt_scale  = *(scale + adjIndex);
        pvt_dscale = (_FLOAT_PREC)0.;
        pvt_dbias  = (_FLOAT_PREC)0.;
        dxhat      = (_FLOAT_PREC)0.;
        dxhathat   = (_FLOAT_PREC)0.;

        for(int n = 0; n < MIO_BN_N; n++)
        {
            // per (x-dims) channel load a block of data into LDS
            index  = in_nstride * n + adjIndex;
            xhat   = (FLOAT2FLOATPREC(in[index]) - mean) * invVar;
            dyelem = FLOAT2FLOATPREC(dy_in[index]);
            pvt_dbias += dyelem;
            pvt_dscale = mad(xhat, dyelem, pvt_dscale);
            tmp1       = pvt_scale * dyelem;
            dxhat += tmp1;
            dxhathat = mad(tmp1, xhat, dxhathat);
        } // end for(n)

        for(int n = 0; n < MIO_BN_N; n++)
        {
            index         = in_nstride * n + adjIndex;
            xhat          = (FLOAT2FLOATPREC(in[index]) - mean) * invVar;
            tmp1          = mad(xhat, dxhathat, dxhat);
            tmp2          = mad((_FLOAT_PREC)N, FLOAT2FLOATPREC(dy_in[index]) * pvt_scale, -tmp1);
            tmp3          = invVar / ((_FLOAT_PREC)N);
            dx_out[index] = FLOATPREC2FLOAT(tmp3 * tmp2);
        }
        // Write out data
        delta_bias[adjIndex]  = pvt_dbias;
        delta_scale[adjIndex] = pvt_dscale;
    } // end for(idx) //image mini_batch is processed
}

//================== END PER ACTIVATION ====================

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
