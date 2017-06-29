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

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

__kernel void BatchNormFwdInferPerActivationEst(
    const __global _FLOAT* in, /* x input */
    unsigned int N,
    unsigned int in_nstride,            /* C*H*W */
    unsigned int in_cstride,            /* H*W */
    __global _FLOAT* out,               /* y output */
    const __global _FLOAT* scale,       /* gamma 1xCxHxW */
    const __global _FLOAT* bias,        /* beta 1xCxHxW */
    __global _FLOAT* estimatedMean,     /*input and output, same descriptor as bias*/
    __global _FLOAT* estimatedVariance, /*input and output*/
    double epsilon)
{

    // PER ACTIVATION
    _FLOAT mean, variance;
    _FLOAT invVariance, elemStd, inhat;
    _FLOAT pvt_scale, pvt_bias;
    unsigned int adjIndex, inImgIndex, index;

    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int yglb_sz = get_global_size(1);

    int Cidx = in_cstride * xgid;

    // move across the sections of an image in the mini_batch stack
    for(int img_offset = 0; img_offset < in_cstride; img_offset += yglb_sz)
    {
        inImgIndex = img_offset + ygid;
        if(inImgIndex < in_cstride)
        {
            adjIndex    = Cidx + inImgIndex; // gamma and beta tensor index
            mean        = estimatedMean[adjIndex];
            variance    = estimatedVariance[adjIndex];
            invVariance = rsqrt(fabs(variance + epsilon));
            pvt_scale   = scale[adjIndex];
            pvt_bias    = bias[adjIndex];
            for(int n = 0; n < N; n++)
            {
                // per (x-dims) channel load a block of data into LDS
                index      = in_nstride * n + adjIndex;
                elemStd    = in[index] - mean; // (x_i - mean)
                inhat      = elemStd * invVariance;
                out[index] = mad(pvt_scale, inhat, pvt_bias); //	y_i = gamma*x_hat + beta
            }                                                 // end for
        }                                                     // end if
    } // end for(img_offset) //image mini_batch is processed
}

//=========================================================

__kernel void BatchNormFwdInferPerActivation(const __global _FLOAT* in, /* x input */
                                             unsigned int N,
                                             unsigned int in_nstride,      /* C*H*W */
                                             unsigned int in_cstride,      /* H*W */
                                             __global _FLOAT* out,         /* y output */
                                             const __global _FLOAT* scale, /* gamma 1xCxHxW */
                                             const __global _FLOAT* bias,  /* beta 1xCxHxW */
                                             double epsilon /* input fuzz param > 0 */
                                             )
{

    // C*H*W is also stored as in_nstride, H*W is in_cstride, W is in_hstride.
    _FLOAT mean_accum, elemStd, variance_accum;
    _FLOAT invVariance, inhat;
    _FLOAT pvt_scale, pvt_bias;

    int xgid = get_global_id(0);
    int ygid = get_global_id(1);

    int yglb_sz = get_global_size(1);

    int Cidx = in_cstride * xgid;

    unsigned int adjIndex, inImgIndex, index;

    // move across the sections of the image mini_batch stack
    for(int img_offset = 0; img_offset < in_cstride; img_offset += yglb_sz)
    {
        mean_accum = 0.;
        inImgIndex = ygid + img_offset;

        // #1 calculate the mean
        // iterating through the stack of images in the mini_batch
        if(inImgIndex < in_cstride)
        {

            adjIndex = Cidx + inImgIndex; // gamma and beta tensor index
            for(int n = 0; n < N; n++)
            {
                index = in_nstride * n + adjIndex;
                mean_accum += in[index];
            } // end for(n)
            mean_accum /= (_FLOAT)N;

            elemStd        = 0.;
            variance_accum = 0.;
            // #2 calculate the variances
            // sigma^2 = (1/batch_mean) * sum( (x_i - batch_mean)^2 )
            for(int n = 0; n < N; n++)
            {
                // per (x-dims) channel load a block of data into LDS
                index   = in_nstride * n + adjIndex;
                elemStd = in[index] - mean_accum; // (x_i - mean) //this is reused but needs recalc
                variance_accum = mad(elemStd, elemStd, variance_accum); // sum{ (x_i - mean)^2 }
            }                                                           // end for(n)
            variance_accum /= (_FLOAT)N; // (1/N)*sum{ (x_i - mean)^2 }

            // #3 add epsilon for numeric stability, sqr_root, and invert
            invVariance = rsqrt(fabs(variance_accum + epsilon));

            // #4 apply the normalization
            // x_hat = (x_i - mean) / sqrt(variance_accum - epsilon)
            pvt_scale = scale[adjIndex];
            pvt_bias  = bias[adjIndex];
            for(int n = 0; n < N; n++)
            {
                // per (x-dims) channel load a block of data into LDS
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
