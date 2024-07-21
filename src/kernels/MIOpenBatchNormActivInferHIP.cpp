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

#include "activation_functions.hpp"

__device__ __forceinline__ void BatchNormFunctionSpatial(const unsigned int n,
                              _FLOAT_PREC* out,
                              const _FLOAT* in,
                              const _FLOAT_PREC mean,
                              const _FLOAT_PREC invVariance,
                              const _FLOAT_PREC scale,
                              const _FLOAT_PREC bias)
{
    for(unsigned int i = 0; i < n; ++i)
    {
        out[i] = fma(scale, ((_FLOAT_PREC)in[i] - mean) * invVariance, bias);
    }
}

extern "C" __global__ __launch_bounds__(MIO_BN_GRP0 * MIO_BN_GRP1 * MIO_BN_GRP2) void
MIOpenBatchNormActivInferSpatialEstHIP(const _FLOAT alpha,
                                    const _FLOAT beta,
                                    const _FLOAT gamma,
                                    double epsilon,
                                    const  _FLOAT* __restrict in,
                                     _FLOAT* __restrict out,
                                    const  _FLOAT_PREC* __restrict bias,
                                    const  _FLOAT_PREC* __restrict scale,
                                    const  _FLOAT_PREC* __restrict estimatedMean,
                                    const  float* __restrict estimatedVariance)
{
    size_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if(blockDim.x != 256){
        printf("gridDim.x = %d, gridDim.y = %d, gridDim.z = %d, blockDim.x = %d, blockDim.y = %d, blockDim.z = %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);    
    }

    __shared__ _FLOAT_PREC lmean;
    __shared__ _FLOAT_PREC lvar;
    __shared__ _FLOAT_PREC lscale;
    __shared__ _FLOAT_PREC lbias;

    int c_i  = tidy;
    int hw_i = tidx;

    unsigned int c_offset = c_i * MIO_BN_HW;

    if(threadIdx.x == 0)
    {
        lmean  = *(estimatedMean + c_i);
        lvar   = *(estimatedVariance + c_i);
        lscale = *(scale + c_i);
        lbias  = *(bias + c_i);
    }
    __syncthreads();

    _FLOAT_PREC pmean  = lmean;
    _FLOAT_PREC pvar   = lvar;
    _FLOAT_PREC pscale = lscale;
    _FLOAT_PREC pbias  = lbias;

    _FLOAT data[MIOPEN_READ_UNIT];
    _FLOAT_PREC invVariance = rsqrt(pvar + epsilon);

     for(unsigned int n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        unsigned int index = n_i * MIO_BN_CHW + c_offset + hw_i * MIOPEN_READ_UNIT;
        *(reinterpret_cast<MIOPEN_READ_TYPE*>(data)) = *(reinterpret_cast<const MIOPEN_READ_TYPE*>(in + index));
        _FLOAT_PREC bnRes[MIOPEN_READ_UNIT];
        _FLOAT_PREC actRes[MIOPEN_READ_UNIT];
        BatchNormFunctionSpatial(MIOPEN_READ_UNIT, bnRes, data, pmean, invVariance, pscale, pbias);
        ActivationFunction(MIOPEN_READ_UNIT, actRes, bnRes, gamma, beta, alpha);
        *(reinterpret_cast<MIOPEN_READ_TYPE*>(out + index)) = *(reinterpret_cast<const MIOPEN_READ_TYPE*>(actRes));

    }
}

extern "C" __global__ void
MIOpenBatchNormActivInferPerActEstHIP(const _FLOAT alpha,
                                   const _FLOAT beta,
                                   const _FLOAT gamma,
                                   double epsilon,
                                   const  _FLOAT* in,
                                    _FLOAT* __restrict out,
                                    const  _FLOAT_PREC* __restrict bias,
                                    const  _FLOAT_PREC* __restrict scale,
                                    const  _FLOAT_PREC* __restrict estimatedMean,
                                    const  _FLOAT_PREC* __restrict estimatedVariance)
{
    unsigned int tidx  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int chw_i = tidx * MIOPEN_READ_UNIT;

    _FLOAT_PREC pmean[MIOPEN_READ_UNIT];
    _FLOAT_PREC pvar[MIOPEN_READ_UNIT];
    _FLOAT_PREC pscale[MIOPEN_READ_UNIT];
    _FLOAT_PREC pbias[MIOPEN_READ_UNIT];

    // Perform a vectorized load of the mean, variance, scale, and bias
    *(reinterpret_cast<MIOPEN_READ_TYPE*>(pmean)) = *(reinterpret_cast<const MIOPEN_READ_TYPE*>(estimatedMean + chw_i));
    *(reinterpret_cast<MIOPEN_READ_TYPE*>(pvar)) = *(reinterpret_cast<const MIOPEN_READ_TYPE*>(estimatedVariance + chw_i));
    *(reinterpret_cast<MIOPEN_READ_TYPE*>(pscale)) = *(reinterpret_cast<const MIOPEN_READ_TYPE*>(scale + chw_i));
    *(reinterpret_cast<MIOPEN_READ_TYPE*>(pbias)) = *(reinterpret_cast<const MIOPEN_READ_TYPE*>(bias + chw_i));

    _FLOAT data[MIOPEN_READ_UNIT];
    _FLOAT_PREC invVariance[MIOPEN_READ_UNIT];

#pragma unroll
    for(unsigned int i = 0; i < MIOPEN_READ_UNIT; i++)
        invVariance[i] = rsqrt((_FLOAT_PREC)pvar[i] + epsilon);

#pragma unroll 2
    for(unsigned int n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        int index                  = n_i * MIO_BN_CHW + chw_i;
        // Perform a vectorized load of the input data
        *(reinterpret_cast<MIOPEN_READ_TYPE*>(data)) = *(reinterpret_cast<const MIOPEN_READ_TYPE*>(in + index));
        _FLOAT_PREC bnRes[MIOPEN_READ_UNIT];
        _FLOAT_PREC actRes[MIOPEN_READ_UNIT];

#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            bnRes[i] = fmaf(pscale[i], ((_FLOAT_PREC)data[i] - pmean[i]) * invVariance[i], pbias[i]);
        }

        ActivationFunction(MIOPEN_READ_UNIT, actRes, bnRes, gamma, beta, alpha);

         *(reinterpret_cast<MIOPEN_READ_TYPE*>(out + index)) = *(reinterpret_cast<const MIOPEN_READ_TYPE*>(actRes));

    }
}