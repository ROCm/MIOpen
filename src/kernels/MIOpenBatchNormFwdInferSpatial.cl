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

#ifndef MIO_BN_NGRPS
#define MIO_BN_NGRPS 1
#endif

#define UNUSED __attribute__((__unused__))

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdInferSpatialEst(const __global _FLOAT* __restrict in, /* x input */
                                  __global _FLOAT* __restrict out,      /* y output */
                                  const __global _FLOAT_PREC* __restrict estimatedMean,
                                  const __global _FLOAT_PREC* __restrict estimatedVariance,
                                  const __global _FLOAT_PREC* __restrict scale,
                                  const __global _FLOAT_PREC* __restrict bias,
                                  double epsilon)
{

    int xgid = get_global_id(0);
    int ygid = get_global_id(1);
    local _FLOAT_PREC lmean;
    local _FLOAT_PREC lvar;
    local _FLOAT_PREC lscale;
    local _FLOAT_PREC lbias;

    unsigned int cidx = xgid * MIO_BN_HW;
    unsigned int index;

    _FLOAT_PREC mean, variance, invVariance;
    _FLOAT_PREC inhat;
    _FLOAT_PREC pscale, pbias;

    if(get_local_id(1) == 0)
    {
        lmean  = *(estimatedMean + xgid);
        lvar   = *(estimatedVariance + xgid);
        lscale = *(scale + xgid); // dims 1xCx1x1
        lbias  = *(bias + xgid);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // move across the sections of the image mini_batch stack
    if(ygid < MIO_BN_HW)
    {
        mean        = lmean;
        variance    = lvar;
        pscale      = lscale;
        pbias       = lbias;
        invVariance = rsqrt(fabs(variance + epsilon));

#pragma unroll
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index      = n * MIO_BN_CHW + cidx + ygid;
            inhat      = ((_FLOAT_PREC)(*(in + index)) - mean) * invVariance;
            out[index] = (_FLOAT)(mad(pscale, inhat, pbias)); // y_i = gamma*x_hat + beta
        }                                                     // end for(img_offset)
    }
} // end spatial norm

#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
