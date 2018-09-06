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

#ifndef MIO_BN_NCHW
#define MIO_BN_NCHW 1
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
#if(MIO_RUNNING_RESULT == 1)
    _FLOAT_PREC pvt_runMean;
    _FLOAT_PREC pvt_newRunMean;
#endif
    unsigned int xgid    = get_global_id(0);
    unsigned int ygid    = get_global_id(1);
    unsigned int yglb_sz = get_global_size(1);
    unsigned int Cidx    = MIO_BN_HW * xgid;
    unsigned int adjIndex, inImgIndex, index;

    _FLOAT_PREC N = (_FLOAT_PREC)MIO_BN_N;

    // move across the sections of the image mini_batch stack
    for(unsigned int img_offset = 0; img_offset < in_cstride; img_offset += yglb_sz)
    {
        inImgIndex = img_offset + ygid;
        if(inImgIndex < in_cstride)
        {
            mean     = (_FLOAT_PREC)0.;
            variance = (_FLOAT_PREC)0.;
            adjIndex = Cidx + inImgIndex; // gamma and beta tensor index

#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            {
                index           = in_nstride * n + adjIndex;
                _FLOAT_PREC xin = (_FLOAT_PREC)(*(in + index));
                mean += xin;
                variance = mad(xin, xin, variance);
            } // end for(n)
            mean /= N;
            variance /= N;
            variance    = mad(-mean, mean, variance);
            invVariance = rsqrt(variance + epsilon);
            pvt_scale   = *(scale + adjIndex);
            pvt_bias    = *(bias + adjIndex);

#if(MIO_SAVE_MEAN_VARIANCE == 1)
            resultSaveInvVariance[adjIndex] = invVariance; /*output only*/
            resultSaveMean[adjIndex]        = mean;
#endif
#if(MIO_RUNNING_RESULT == 1)
            pvt_runMean    = *(resultRunningMean + adjIndex); // previous: oldRunMean
            pvt_newRunMean = mad((_FLOAT_PREC)-expAvgFactor,
                                 pvt_runMean,
                                 pvt_runMean); // tmp = oldRunMean*(1-factor)

            resultRunningMean[adjIndex] = mad((_FLOAT_PREC)mean,
                                              (_FLOAT_PREC)expAvgFactor,
                                              pvt_newRunMean); // newMean*factor + tmp

            const _FLOAT_PREC adjust = (MIO_BN_N == 1) ? variance : variance * (N / (N - 1));
            resultRunningVariance[adjIndex] =
                (1 - (_FLOAT_PREC)expAvgFactor) * *(resultRunningVariance + adjIndex) +
                (_FLOAT_PREC)expAvgFactor * adjust;
#endif

#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            { // per (x-dims) channel load a block of data unsigned into LDS
                index      = in_nstride * n + adjIndex;
                inhat      = ((_FLOAT_PREC)(*(in + index)) - mean) * invVariance;
                out[index] = (_FLOAT)(mad(pvt_scale, inhat, pvt_bias));
            } // end for(n)
        }     // end if(inImgIndex)
    }         // end for(img_offset) //image mini_batch is processed
}

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
