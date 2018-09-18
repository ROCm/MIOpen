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

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
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

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdInferPerActivationEst(
    const __global _FLOAT* in,                      /* x input */
    __global _FLOAT* __restrict out,                /* y output */
    __global _FLOAT_PREC* __restrict estimatedMean, /*input and output, same descriptor as bias*/
    __global _FLOAT_PREC* __restrict estimatedVariance, /*input and output*/
    const __global _FLOAT_PREC* __restrict scale,       /* gamma 1xCxHxW */
    const __global _FLOAT_PREC* __restrict bias,        /* beta 1xCxHxW */
    double epsilon)
{

    // PER ACTIVATION
    _FLOAT_PREC mean, variance;
    _FLOAT_PREC invVariance, elemStd, inhat;
    _FLOAT_PREC pvt_scale, pvt_bias;
    unsigned int adjIndex, inImgIndex, index;

    int xgid    = get_global_id(0);
    int ygid    = get_global_id(1);
    int yglb_sz = get_global_size(1);

    int Cidx = MIO_BN_HW * xgid;

    // move across the sections of an image in the mini_batch stack
    for(int img_offset = 0; img_offset < MIO_BN_HW; img_offset += yglb_sz)
    {
        inImgIndex = img_offset + ygid;
        if(inImgIndex < MIO_BN_HW)
        {
            adjIndex    = Cidx + inImgIndex; // gamma and beta tensor index
            mean        = estimatedMean[adjIndex];
            variance    = estimatedVariance[adjIndex];
            invVariance = rsqrt(fabs(variance + epsilon));
            pvt_scale   = *(scale + adjIndex);
            pvt_bias    = *(bias + adjIndex);

#pragma unroll
            for(int n = 0; n < MIO_BN_N; n++)
            {
                // per (x-dims) channel load a block of data into LDS
                index   = MIO_BN_CHW * n + adjIndex;
                elemStd = (_FLOAT_PREC)(*(in + index)) - mean; // (x_i - mean)
                inhat   = elemStd * invVariance;
                out[index] =
                    (_FLOAT)(mad(pvt_scale, inhat, pvt_bias)); //	y_i = gamma*x_hat + beta
            }                                                  // end for
        }                                                      // end if
    } // end for(img_offset) //image mini_batch is processed
}

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
