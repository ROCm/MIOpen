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

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

//==================== PER ACTIVATION =======================

__kernel void BatchNormFwdTrainPerActivation(
    const __global _FLOAT* __restrict in,    /* x input */
    unsigned int in_nstride,                 /* C*H*W */
    unsigned int in_cstride,                 /* H*W */
    __global _FLOAT* __restrict out,         /* y output */
    const __global _FLOAT* __restrict scale, /* gamma 1xCxHxW */
    const __global _FLOAT* __restrict bias,  /* beta 1xCxHxW */
    double expAvgFactor,                     /* input momentum */
#if(MIO_RUNNING_RESULT == 1)
    __global _FLOAT* __restrict resultRunningMean,     /*input and output, same descriptor as bias*/
    __global _FLOAT* __restrict resultRunningVariance, /*input and output*/
#endif
    double epsilon /* input fuzz param > 0 */
#if(MIO_SAVE_MEAN_VARIANCE == 1)
    ,
    __global _FLOAT* __restrict resultSaveMean,       /*output only*/
    __global _FLOAT* __restrict resultSaveInvVariance /*output only*/
#endif
    )
{

    // PER ACTIVATION
    _FLOAT mean_accum, variance_accum = expAvgFactor;
    _FLOAT elemStd;
    _FLOAT invVariance, inhat;
    _FLOAT pvt_scale, pvt_bias;
#if(MIO_RUNNING_RESULT == 1)
    _FLOAT pvt_runMean;
    _FLOAT pvt_newRunMean;
#endif
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int yglb_sz = get_global_size(1);

    unsigned int Cidx = MIO_BN_HW * xgid;
    unsigned int adjIndex, inImgIndex, index;

    _FLOAT N = (_FLOAT)MIO_BN_N;

    // move across the sections of the image mini_batch stack
    for(unsigned int img_offset = 0; img_offset < in_cstride; img_offset += yglb_sz)
    {

        inImgIndex = img_offset + ygid;
        // #1 calculate the mean
        // iterating through the stack of images in the mini_batch
        if(inImgIndex < in_cstride)
        {
            mean_accum = 0.;
            adjIndex   = Cidx + inImgIndex; // gamma and beta tensor index

#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            {
                index = in_nstride * n + adjIndex;
                mean_accum += in[index];
            } // end for(n)
            mean_accum /= N;
#if(MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveMean[adjIndex] = mean_accum;
#endif
/*
Note from cuDNN: expAvgFactor
    Factor used in the moving average computation:
    runningMean = newMean*factor + runningMean*(1-factor).
    Use a factor=1/(1+n) at N-th call to the function to get
    Cumulative Moving Average (CMA) behavior:
            CMA[n] = (x[1]+...+x[n])/n.
    Since CMA[n+1]
            = (n*CMA[n]+x[n+1])/(n+1)
            = ((n+1)*CMA[n]-CMA[n])/(n+1) + x[n+1]/(n+1)
            = CMA[n]*(1-1/(n+1))+x[n+1]*1/(n+1)
*/
#if(MIO_RUNNING_RESULT == 1)
            pvt_runMean = resultRunningMean[adjIndex]; // previous: oldRunMean
            pvt_newRunMean =
                mad((_FLOAT)-expAvgFactor, pvt_runMean, pvt_runMean); // tmp = oldRunMean*(1-factor)
            resultRunningMean[adjIndex] = mad(
                (_FLOAT)mean_accum, (_FLOAT)expAvgFactor, pvt_newRunMean); // newMean*factor + tmp
#endif

            elemStd        = 0.;
            variance_accum = 0.;
// #2 calculate the variances
// sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )

#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            {
                // per (x-dims) channel load a block of data unsigned into LDS
                index   = in_nstride * n + adjIndex;
                elemStd = in[index] - mean_accum; // (x_i - mean) //this is reused but needs recalc
                variance_accum =
                    mad(elemStd, elemStd, (_FLOAT)variance_accum); // sum{ (x_i - mean)^2 }
            }                                                      // end for(n)
            variance_accum /= N;                                   // (1/N)*sum{ (x_i - mean)^2 }

#if(MIO_RUNNING_RESULT == 1)
            const _FLOAT adjust = (MIO_BN_N == 1) ? variance_accum : variance_accum * (N / (N - 1));
            const _FLOAT tmp    = mad((_FLOAT)-expAvgFactor, adjust, adjust);
            resultRunningVariance[adjIndex] =
                mad((_FLOAT)expAvgFactor, resultRunningVariance[adjIndex], tmp);
#endif

            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVariance = rsqrt(variance_accum + epsilon);

#if(MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveInvVariance[adjIndex] = invVariance; /*output only*/
#endif

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
            pvt_scale = scale[adjIndex];
            pvt_bias  = bias[adjIndex];

#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            {
                // per (x-dims) channel load a block of data unsigned into LDS
                index   = in_nstride * n + adjIndex;
                elemStd = in[index] - mean_accum; // (x_i - mean)
                inhat   = elemStd * invVariance;
                // #5 Gamma and Beta adjust
                //	y_i = gamma*x_hat + beta
                out[index] = mad(pvt_scale, inhat, pvt_bias);
            } // end for(n)
        }     // end if(inImgIndex)
    }         // end for(img_offset) //image mini_batch is processed
}

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
