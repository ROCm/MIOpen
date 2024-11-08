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

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdInferSpatialEst(const __global _FLOAT* __restrict in, /* x input */
                                  __global _FLOAT* __restrict out,      /* y output */
                                  const __global _FLOAT_PREC* __restrict estimatedMean,
                                  const __global _FLOAT_PREC* __restrict estimatedVariance,
                                  const __global _FLOAT_PREC* __restrict scale,
                                  const __global _FLOAT_PREC* __restrict bias,
                                  double epsilon,
                                  unsigned int batchSize,
                                  unsigned int imageDims,
                                  unsigned int batchStride)
{

    int xgid = get_global_id(0);
    int ygid = get_global_id(1);

    unsigned int index;

    _FLOAT_PREC mean, variance, invVariance;
    _FLOAT_PREC inhat;
    _FLOAT_PREC pscale, pbias;

    mean        = *(estimatedMean + xgid);
    variance    = *(estimatedVariance + xgid);
    pscale      = *(scale + xgid);
    pbias       = *(bias + xgid);
    invVariance = rsqrt(fabs(variance + epsilon));

    for(int idx = ygid; idx < imageDims; idx += get_global_size(1))
    {
        for(int n = 0; n < batchSize; n++)
        {
            index      = (n * batchStride) + (xgid * imageDims) + idx;
            inhat      = (FLOAT2FLOATPREC(*(in + index)) - mean) * invVariance;
            out[index] = FLOATPREC2FLOAT(mad(pscale, inhat, pbias));
        }
    }
} // end spatial norm

#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
