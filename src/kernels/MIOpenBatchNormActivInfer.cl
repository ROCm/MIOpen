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
#include "activation_functions.h"

void BatchNormFunctionSpatial(const uint n,
                              _FLOAT_PREC* out,
                              const _FLOAT* in,
                              const _FLOAT_PREC mean,
                              const _FLOAT_PREC invVariance,
                              const _FLOAT_PREC scale,
                              const _FLOAT_PREC bias)
{
    for(uint i = 0; i < n; ++i)
    {
        out[i] = mad(scale, (FLOAT2FLOATPREC(in[i]) - mean) * invVariance, bias);
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormActivInferSpatialEst(const _FLOAT alpha,
                                    const _FLOAT beta,
                                    const _FLOAT gamma,
                                    double epsilon,
                                    const __global _FLOAT* __restrict in,
                                    __global _FLOAT* __restrict out,
                                    const __global _FLOAT_PREC* __restrict bias,
                                    const __global _FLOAT_PREC* __restrict scale,
                                    const __global _FLOAT_PREC* __restrict estimatedMean,
                                    const __global _FLOAT_PREC* __restrict estimatedVariance)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);

    __local _FLOAT_PREC lmean;
    __local _FLOAT_PREC lvar;
    __local _FLOAT_PREC lscale;
    __local _FLOAT_PREC lbias;

    int c_i  = gid1;
    int hw_i = gid0;

    unsigned int c_offset = c_i * MIO_BN_HW;

    if(get_local_id(0) == 0)
    {
        lmean  = *(estimatedMean + c_i);
        lvar   = *(estimatedVariance + c_i);
        lscale = *(scale + c_i);
        lbias  = *(bias + c_i);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    _FLOAT_PREC pmean  = lmean;
    _FLOAT_PREC pvar   = lvar;
    _FLOAT_PREC pscale = lscale;
    _FLOAT_PREC pbias  = lbias;

    _FLOAT data[MIOPEN_READ_UNIT];
    _FLOAT_PREC invVariance = rsqrt(pvar + epsilon);

    __attribute__((opencl_unroll_hint(2))) for(unsigned int n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        int index = n_i * MIO_BN_CHW + c_offset + hw_i * MIOPEN_READ_UNIT;

        *((MIOPEN_READ_TYPE*)data) = *((const __global MIOPEN_READ_TYPE*)(in + index));
        _FLOAT_PREC bnRes[MIOPEN_READ_UNIT];
        _FLOAT_PREC actRes[MIOPEN_READ_UNIT];
        BatchNormFunctionSpatial(MIOPEN_READ_UNIT, bnRes, data, pmean, invVariance, pscale, pbias);
        ActivationFunction(MIOPEN_READ_UNIT,
                           actRes,
                           bnRes,
                           FLOAT2FLOATPREC(gamma),
                           FLOAT2FLOATPREC(beta),
                           FLOAT2FLOATPREC(alpha));
        for(int i = 0; i < MIOPEN_READ_UNIT; i++)
        {
            out[index + i] = FLOATPREC2FLOAT(actRes[i]);
        }
    }
} // end spatial norm

void BatchNormFunctionPerAct(const uint n,
                             _FLOAT_PREC* out,
                             const _FLOAT* in,
                             const _FLOAT_PREC* mean,
                             const _FLOAT_PREC* invVariance,
                             const _FLOAT_PREC* scale,
                             const _FLOAT_PREC* bias)
{
    for(uint i = 0; i < n; ++i)
    {
        out[i] = mad(scale[i], (FLOAT2FLOATPREC(in[i]) - mean[i]) * invVariance[i], bias[i]);
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormActivInferPerActEst(const _FLOAT alpha,
                                   const _FLOAT beta,
                                   const _FLOAT gamma,
                                   double epsilon,
                                   const __global _FLOAT* in,
                                   __global _FLOAT* __restrict out,
                                   const __global _FLOAT_PREC* __restrict bias,
                                   const __global _FLOAT_PREC* __restrict scale,
                                   const __global _FLOAT_PREC* __restrict estimatedMean,
                                   const __global _FLOAT_PREC* __restrict estimatedVariance)
{
    int gid0  = get_global_id(0);
    int chw_i = gid0 * MIOPEN_READ_UNIT;

    _FLOAT_PREC pmean[MIOPEN_READ_UNIT];
    _FLOAT_PREC pvar[MIOPEN_READ_UNIT];
    _FLOAT_PREC pscale[MIOPEN_READ_UNIT];
    _FLOAT_PREC pbias[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)pmean)  = *((const __global MIOPEN_READ_TYPE*)(estimatedMean + chw_i));
    *((MIOPEN_READ_TYPE*)pvar)   = *((const __global MIOPEN_READ_TYPE*)(estimatedVariance + chw_i));
    *((MIOPEN_READ_TYPE*)pscale) = *((const __global MIOPEN_READ_TYPE*)(scale + chw_i));
    *((MIOPEN_READ_TYPE*)pbias)  = *((const __global MIOPEN_READ_TYPE*)(bias + chw_i));
    for(int i = 0; i < MIOPEN_READ_UNIT; i++)
    {
        pmean[i]  = estimatedMean[chw_i + i];
        pvar[i]   = estimatedVariance[chw_i + i];
        pscale[i] = scale[chw_i + i];
        pbias[i]  = bias[chw_i + i];
    }
    _FLOAT data[MIOPEN_READ_UNIT];
    _FLOAT_PREC invVariance[MIOPEN_READ_UNIT];

    for(int i = 0; i < MIOPEN_READ_UNIT; i++)
        invVariance[i] = rsqrt(pvar[i] + epsilon);

    __attribute__((opencl_unroll_hint(2))) for(uint n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        int index                  = n_i * MIO_BN_CHW + chw_i;
        *((MIOPEN_READ_TYPE*)data) = *((const __global MIOPEN_READ_TYPE*)(in + index));
        _FLOAT_PREC bnRes[MIOPEN_READ_UNIT];
        _FLOAT_PREC actRes[MIOPEN_READ_UNIT];
        BatchNormFunctionPerAct(MIOPEN_READ_UNIT, bnRes, data, pmean, invVariance, pscale, pbias);
        ActivationFunction(MIOPEN_READ_UNIT,
                           actRes,
                           bnRes,
                           FLOAT2FLOATPREC(gamma),
                           FLOAT2FLOATPREC(beta),
                           FLOAT2FLOATPREC(alpha));
        for(int i = 0; i < MIOPEN_READ_UNIT; i++)
        {
            out[index + i] = FLOATPREC2FLOAT(actRes[i]);
        }
    }
}

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif // #ifdef LITE
