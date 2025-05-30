/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
#include "bnorm_spatial_activation_functions.h"

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdInferSpatialEst(const __global _FLOAT* __restrict in, /* x input */
                                  __global _FLOAT* __restrict out,      /* y output */
                                  const __global _FLOAT_PREC* __restrict estimatedMean,
                                  const __global _FLOAT_PREC* __restrict estimatedVariance,
                                  const __global _FLOAT_PREC* __restrict scale,
                                  const __global _FLOAT_PREC* __restrict bias,
                                  double epsilon,
                                  unsigned int c,
                                  unsigned int hw,
                                  unsigned int batchSize,
                                  unsigned int cStride,
                                  unsigned int hwStride,
                                  unsigned int batchStride,
                                  _FLOAT_PREC _alpha,
                                  _FLOAT_PREC _beta)
{

    ACTIVATION_SET()
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);

    if(xgid * VEC_SIZE_X >= c || ygid * VEC_SIZE_Y >= hw)
        return;

    unsigned int index;
    _FLOAT_PREC_C mean, variance, invVariance;
    _FLOAT_PREC_C pscale, pbias;
    _FLOAT_PREC_LS inhat;
    _FLOAT_LS value;

    mean        = *((const __global _FLOAT_PREC_C*)(estimatedMean + xgid * VEC_SIZE_X));
    variance    = *((const __global _FLOAT_PREC_C*)(estimatedVariance + xgid * VEC_SIZE_X));
    pscale      = *((const __global _FLOAT_PREC_C*)(scale + xgid * VEC_SIZE_X));
    pbias       = *((const __global _FLOAT_PREC_C*)(bias + xgid * VEC_SIZE_X));
    invVariance = rsqrt(fabs(variance + (_FLOAT_PREC_C)epsilon));

    for(int n = 0; n < batchSize; n++)
    {
        index = (n * batchStride) + (xgid * cStride * VEC_SIZE_X) + (ygid * hwStride * VEC_SIZE_Y);
        value = *((const __global _FLOAT_LS*)(in + index));

        inhat = FLOAT2FLOATPREC_VEC(value);
        inhat = (inhat - mean) * invVariance;
        inhat = mad(pscale, inhat, (_FLOAT_PREC_LS)pbias);
        ACTIVATION_OP(inhat, inhat, _FLOAT_PREC_LS)
        value = FLOATPREC2FLOAT_VEC(inhat);

        *((__global _FLOAT_LS*)(out + index)) = value;
    }
} // end spatial norm

#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
