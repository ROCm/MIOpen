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

//==================== PER ACTIVATION =======================

__kernel void MIOpenBatchNormFwdTrainPerActivation(
    const __global _FLOAT* __restrict in,         /* x input */
    unsigned int in_nstride,                      /* C*H*W */
    unsigned int in_cstride,                      /* H*W */
    __global _FLOAT* __restrict out,              /* y output */
    const __global _FLOAT_PREC* __restrict scale, /* gamma 1xCxHxW */
    const __global _FLOAT_PREC* __restrict bias,  /* beta 1xCxHxW */
#if(MIO_RUNNING_RESULT == 1)
    double expAvgFactor, /* input momentum */
    __global
        _FLOAT_PREC* __restrict resultRunningMean, /*input and output, same descriptor as bias*/
    __global _FLOAT_PREC* __restrict resultRunningVariance, /*input and output*/
#endif
    double epsilon /* input fuzz param > 0 */
#if(MIO_SAVE_MEAN_VARIANCE == 1)
    ,
    __global _FLOAT_PREC* __restrict resultSaveMean,       /*output only*/
    __global _FLOAT_PREC* __restrict resultSaveInvVariance /*output only*/
#endif
)
{

    // PER ACTIVATION
    _FLOAT_PREC mean        = 0.;
    _FLOAT_PREC variance    = 0.;
    _FLOAT_PREC invVariance = 0.;
    _FLOAT_PREC inhat       = 0.;
    _FLOAT_PREC pvt_scale   = 0.;
    _FLOAT_PREC pvt_bias    = 0.;

    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int yglb_sz = get_global_size(1);
    int cidx             = MIO_BN_HW * xgid;
    int adjIndex, index;

    _FLOAT_PREC invN = 1.0 / (_FLOAT_PREC)MIO_BN_N;

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
        mean *= invN;
        variance = 0.;

        for(int n = 0; n < MIO_BN_N; n++)
        {
            index             = in_nstride * n + adjIndex;
            _FLOAT_PREC xdiff = FLOAT2FLOATPREC(in[index]) - mean;
            variance += (xdiff * xdiff);
        } // end for(n)
        variance *= invN;
        invVariance = rsqrt(variance + epsilon);
        pvt_scale   = *(scale + adjIndex);
        pvt_bias    = *(bias + adjIndex);

#if(MIO_RUNNING_RESULT == 1)
        running_stash_pa(
            resultRunningMean, resultRunningVariance, expAvgFactor, mean, variance, adjIndex);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        saved_stash(resultSaveMean, resultSaveInvVariance, mean, invVariance, adjIndex);
#endif

        for(int n = 0; n < MIO_BN_N; n++)
        { // per (x-dims) channel load a block of data unsigned into LDS
            index      = in_nstride * n + adjIndex;
            inhat      = (FLOAT2FLOATPREC(in[index]) - mean) * invVariance;
            out[index] = FLOATPREC2FLOAT(mad(pvt_scale, inhat, pvt_bias));
        } // end for(n)
    }     // end for(img_offset) //image mini_batch is processed
}

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
