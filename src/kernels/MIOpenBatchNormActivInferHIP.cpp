/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "activation_functions.hpp"

extern "C" __global__ void __launch_bounds__(MIO_BN_GRP0* MIO_BN_GRP1* MIO_BN_GRP2)
    MIOpenBatchNormActivInferSpatialEstHIP(const FP_TYPE_PREC alpha,
                                           const FP_TYPE_PREC beta,
                                           const FP_TYPE_PREC gamma,
                                           const double epsilon,
                                           const FP_TYPE* __restrict in,
                                           FP_TYPE* __restrict out,
                                           const FP_TYPE_PREC* __restrict bias,
                                           const FP_TYPE_PREC* __restrict scale,
                                           const FP_TYPE_PREC* __restrict estimatedMean,
                                           const FP_TYPE_PREC* __restrict estimatedVariance)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    // HIP runtime does not support launching non-uniform blocks
    // So extra threads are launched to handle this.
    if(tidx >= MIOPEN_SBN_BOUNDS)
        return;

    unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int c_i      = tidy;
    unsigned int hw_i     = tidx;
    unsigned int c_offset = c_i * MIO_BN_HW;

    // Load the mean, variance, scale, and bias that is broadcast across the block
    const FP_TYPE_PREC pmean       = estimatedMean[c_i];
    const FP_TYPE_PREC pvar        = estimatedVariance[c_i];
    const FP_TYPE_PREC pscale      = scale[c_i];
    const FP_TYPE_PREC pbias       = bias[c_i];
    const FP_TYPE_PREC invVariance = rsqrt(pvar + epsilon);
    // Load the input data. This is done in a vectorized manner
    FP_TYPE data[MIOPEN_READ_UNIT];

#pragma unroll 2
    for(unsigned int n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        const unsigned int index = n_i * MIO_BN_CHW + c_offset + hw_i * MIOPEN_READ_UNIT;
        // Perform a vectorized load of the input data
        *(reinterpret_cast<MIOPEN_READ_TYPE*>(data)) =
            *(reinterpret_cast<const MIOPEN_READ_TYPE*>(in + index));
        FP_TYPE_PREC bnRes[MIOPEN_READ_UNIT];
        FP_TYPE_PREC actRes[MIOPEN_READ_UNIT];
#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            bnRes[i] =
                fma(pscale, (static_cast<FP_TYPE_PREC>(data[i]) - pmean) * invVariance, pbias);
        }
        ActivationFunction(actRes, bnRes, gamma, beta, alpha);
        if constexpr(MIOPEN_USE_FP16)
        {
#pragma unroll
            for(unsigned int i = 0; i < MIOPEN_READ_UNIT; i++)
            {
                out[index + i] = static_cast<FP_TYPE>(actRes[i]);
            }
        }
        else
        {
            // perform a vectorized store of the output data as FP_TYPE and PF_TYPE_PREC are same
            // for FP32
            *(reinterpret_cast<MIOPEN_READ_TYPE*>(out + index)) =
                *(reinterpret_cast<const MIOPEN_READ_TYPE*>(actRes));
        }
    }
}

extern "C" __global__ void __launch_bounds__(MIO_BN_GRP0* MIO_BN_GRP1* MIO_BN_GRP2)
    MIOpenBatchNormActivInferPerActEstHIP(const FP_TYPE_PREC alpha,
                                          const FP_TYPE_PREC beta,
                                          const FP_TYPE_PREC gamma,
                                          const double epsilon,
                                          const FP_TYPE* __restrict in,
                                          FP_TYPE* __restrict out,
                                          const FP_TYPE_PREC* __restrict bias,
                                          const FP_TYPE_PREC* __restrict scale,
                                          const FP_TYPE_PREC* __restrict estimatedMean,
                                          const FP_TYPE_PREC* __restrict estimatedVariance)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    // HIP runtime does not support launching non-uniform blocks
    // So extra threads are launched to handle this.
    if(tidx >= MIOPEN_SBN_BOUNDS)
        return;

    unsigned int chw_i = tidx * MIOPEN_READ_UNIT;

    FP_TYPE_PREC pmean[MIOPEN_READ_UNIT];
    FP_TYPE_PREC pvar[MIOPEN_READ_UNIT];
    FP_TYPE_PREC pscale[MIOPEN_READ_UNIT];
    FP_TYPE_PREC pbias[MIOPEN_READ_UNIT];

    // Perform a vectorized load of the mean, variance, scale, and bias
    *(reinterpret_cast<MIOPEN_PREC_READ_TYPE*>(pmean)) =
        *(reinterpret_cast<const MIOPEN_PREC_READ_TYPE*>(estimatedMean + chw_i));
    *(reinterpret_cast<MIOPEN_PREC_READ_TYPE*>(pvar)) =
        *(reinterpret_cast<const MIOPEN_PREC_READ_TYPE*>(estimatedVariance + chw_i));
    *(reinterpret_cast<MIOPEN_PREC_READ_TYPE*>(pscale)) =
        *(reinterpret_cast<const MIOPEN_PREC_READ_TYPE*>(scale + chw_i));
    *(reinterpret_cast<MIOPEN_PREC_READ_TYPE*>(pbias)) =
        *(reinterpret_cast<const MIOPEN_PREC_READ_TYPE*>(bias + chw_i));

    FP_TYPE data[MIOPEN_READ_UNIT];
    FP_TYPE_PREC invVariance[MIOPEN_READ_UNIT];

#pragma unroll
    for(unsigned int i = 0; i < MIOPEN_READ_UNIT; i++)
        invVariance[i] = rsqrt(pvar[i] + epsilon);

#pragma unroll 2
    for(unsigned int n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        const unsigned int index = n_i * MIO_BN_CHW + chw_i;
        // Perform a vectorized load of the input data
        *(reinterpret_cast<MIOPEN_READ_TYPE*>(data)) =
            *(reinterpret_cast<const MIOPEN_READ_TYPE*>(in + index));
        FP_TYPE_PREC bnRes[MIOPEN_READ_UNIT];
        FP_TYPE_PREC actRes[MIOPEN_READ_UNIT];
#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            bnRes[i] = fma(pscale[i],
                           (static_cast<FP_TYPE_PREC>(data[i]) - pmean[i]) * invVariance[i],
                           pbias[i]);
        }

        ActivationFunction(actRes, bnRes, gamma, beta, alpha);
        if constexpr(MIOPEN_USE_FP16)
        { // In this situation, FP_TYPE_PREC is FP32 whereas FP_TYPE is FP16
          // So, we cannot perform a vectorized store
#pragma unroll
            for(unsigned int i = 0; i < MIOPEN_READ_UNIT; i++)
            {
                out[index + i] = static_cast<FP_TYPE>(actRes[i]);
            }
        }
        else
        {
            // perform a vectorized store of the output data as FP_TYPE and PF_TYPE_PREC are same
            *(reinterpret_cast<MIOPEN_READ_TYPE*>(out + index)) =
                *(reinterpret_cast<const MIOPEN_READ_TYPE*>(actRes));
        }
    }
}
