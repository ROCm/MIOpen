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

#if(MIOPEN_USE_FP16 == 1 && MIOPEN_USE_FPMIX == 0)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_PREC half
#ifndef HALF_MAX
#define MAX_VAL 65504
#else
#define MAX_VAL HALF_MAX
#endif

#define EPSILON (_FLOAT_PREC)0.0001

#elif(MIOPEN_USE_FP32 == 1 && MIOPEN_USE_FPMIX == 0)
#define _FLOAT float
#define _FLOAT_PREC float
#define EPSILON (_FLOAT)0.000001
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F
#else
#define MAX_VAL FLT_MAX
#endif

#elif MIOPEN_USE_FPMIX == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_PREC float
#define EPSILON (_FLOAT_PREC)0.000001
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#define UNUSED __attribute__((__unused__))

#define MIOPEN_NRN_GROUP_SZ2 1

#define MIOPEN_NEURON_PASTHRU 0      // x
#define MIOPEN_NEURON_LOGISTIC 1     // 1 / (1 + e^-x)	//Sigmoid
#define MIOPEN_NEURON_TANH 2         // beta * tanh(alpha * x)
#define MIOPEN_NEURON_RELU 3         // max(0, x)
#define MIOPEN_NEURON_SOFTRELU 4     // log(1 + e^x)   // bonomial normal log likelihood
#define MIOPEN_NEURON_ABS 5          // abs(x)
#define MIOPEN_NEURON_POWER 6        // (alpha + beta * x )^gamma
#define MIOPEN_NEURON_CLIPPED_RELU 7 // min(alpha, max(0, x))
#define MIOPEN_NEURON_LEAKY_RELU 8   // alpha * x | x <= 0; x | x > 0
#define MIOPEN_NEURON_ELU 9          // alpha * (e^x - 1) | x <= 0; x | x > 0
//#define MIOPEN_NEURON_SQUARE 10      // x^2
//#define MIOPEN_NEURON_SQR 11         // sqr(x)
#define MIOPEN_NEURON_TOTAL 10

__attribute__((always_inline)) uint iDiv(uint v, uint d)
{
    uint r = (uint)((float)v * (1.0f / (float)d) + 0.00001f);
    return (r);
}

__attribute__((always_inline)) uint iMod(uint v, uint u, uint d)
{
    uint r = v - mul24(u, d);
    return (r);
}

static __constant _FLOAT_PREC kBNLL_THRESHOLD = (_FLOAT_PREC)50.;

__attribute__((always_inline)) void ActivationFunction_PassThru(const uint n,
                                                                _FLOAT_PREC* res,
                                                                const _FLOAT_PREC* data,
                                                                UNUSED const _FLOAT_PREC gamma,
                                                                UNUSED const _FLOAT_PREC beta,
                                                                UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = data[i];
    }
}

__attribute__((always_inline)) void ActivationFunction_ReLU(const uint n,
                                                            _FLOAT_PREC* res,
                                                            const _FLOAT_PREC* data,
                                                            UNUSED const _FLOAT_PREC gamma,
                                                            UNUSED const _FLOAT_PREC beta,
                                                            UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = (data[i] > 0) ? data[i] : 0.;
    }
}

__attribute__((always_inline)) void ActivationFunction_Sigmoid(const uint n,
                                                               _FLOAT_PREC* res,
                                                               const _FLOAT_PREC* data,
                                                               UNUSED const _FLOAT_PREC gamma,
                                                               UNUSED const _FLOAT_PREC beta,
                                                               UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = 1/(1 + exp(-x))
        res[i] = (_FLOAT_PREC)1.f / ((_FLOAT_PREC)1.f + exp(-data[i]));
    }
}

__attribute__((always_inline)) void ActivationFunction_TanH(const uint n,
                                                            _FLOAT_PREC* res,
                                                            const _FLOAT_PREC* data,
                                                            UNUSED const _FLOAT_PREC gamma,
                                                            const _FLOAT_PREC beta,
                                                            const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = beta * tanh(alpha * x)
        res[i] = beta * tanh(alpha * data[i]);
    }
}

__attribute__((always_inline)) void ActivationFunction_Abs(const uint n,
                                                           _FLOAT_PREC* res,
                                                           const _FLOAT_PREC* data,
                                                           UNUSED const _FLOAT_PREC gamma,
                                                           UNUSED const _FLOAT_PREC beta,
                                                           UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = fabs(data[i]);
    }
}

__attribute__((always_inline)) void ActivationFunction_Square(const uint n,
                                                              _FLOAT_PREC* res,
                                                              const _FLOAT_PREC* data,
                                                              UNUSED const _FLOAT_PREC gamma,
                                                              UNUSED const _FLOAT_PREC beta,
                                                              UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {

        res[i] = data[i] * data[i];
    }
}

__attribute__((always_inline)) void ActivationFunction_Sqrt(const uint n,
                                                            _FLOAT_PREC* res,
                                                            const _FLOAT_PREC* data,
                                                            UNUSED const _FLOAT_PREC gamma,
                                                            UNUSED const _FLOAT_PREC beta,
                                                            UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {

        res[i] = sqrt(data[i]);
    }
}

__attribute__((always_inline)) void ActivationFunction_Linear(const uint n,
                                                              _FLOAT_PREC* res,
                                                              const _FLOAT_PREC* data,
                                                              UNUSED const _FLOAT_PREC gamma,
                                                              const _FLOAT_PREC beta,
                                                              const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = alpha + beta * data[i];
    }
}

__attribute__((always_inline)) void ActivationFunction_Power(const uint n,
                                                             _FLOAT_PREC* res,
                                                             const _FLOAT_PREC* data,
                                                             const _FLOAT_PREC gamma,
                                                             const _FLOAT_PREC beta,
                                                             const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = (alpha + beta * x ) ^ gamma
        _FLOAT_PREC arg = alpha + data[i] * beta;
        res[i]          = arg <= EPSILON ? (_FLOAT_PREC)0 : pow(arg, gamma);
    }
}

__attribute__((always_inline)) void ActivationFunction_BNLL(const uint n,
                                                            _FLOAT_PREC* res,
                                                            const _FLOAT_PREC* data,
                                                            UNUSED const _FLOAT_PREC gamma,
                                                            UNUSED const _FLOAT_PREC beta,
                                                            UNUSED const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        //  y = log(1 + exp(x))
        res[i] = (data[i] > 0) ? (data[i] + log((_FLOAT_PREC)1.f + exp(-data[i])))
                               : log((_FLOAT_PREC)(1.f) + exp(data[i]));
    }
}

__attribute__((always_inline)) void ActivationFunction_Leaky_ReLU(const uint n,
                                                                  _FLOAT_PREC* res,
                                                                  const _FLOAT_PREC* data,
                                                                  UNUSED const _FLOAT_PREC gamma,
                                                                  UNUSED const _FLOAT_PREC beta,
                                                                  const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = data[i] * ((data[i] > 0) ? (_FLOAT_PREC)1.f : alpha);
    }
}

__attribute__((always_inline)) void ActivationFunction_Clipped_ReLU(const uint n,
                                                                    _FLOAT_PREC* res,
                                                                    const _FLOAT_PREC* data,
                                                                    UNUSED const _FLOAT_PREC gamma,
                                                                    UNUSED const _FLOAT_PREC beta,
                                                                    const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = fmin(alpha, fmax(data[i], 0));
    }
}

__attribute__((always_inline)) void ActivationFunction_ELU(const uint n,
                                                           _FLOAT_PREC* res,
                                                           const _FLOAT_PREC* data,
                                                           UNUSED const _FLOAT_PREC gamma,
                                                           UNUSED const _FLOAT_PREC beta,
                                                           const _FLOAT_PREC alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = (data[i] > 0) ? data[i] : (alpha * (exp(data[i]) - (_FLOAT_PREC)1.f));
    }
}

__attribute__((always_inline)) void ActivationFunction(const uint n,
                                                       _FLOAT_PREC* res,
                                                       const _FLOAT_PREC* data,
                                                       const _FLOAT_PREC gamma,
                                                       const _FLOAT_PREC beta,
                                                       const _FLOAT_PREC alpha)
{
#if MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU
    {
        ActivationFunction_PassThru(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LOGISTIC
    {
        // y = 1/(1 + exp(-x))
        ActivationFunction_Sigmoid(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_TANH
    {
        // y = beta * tanh(alpha * x)
        ActivationFunction_TanH(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU
    {
        ActivationFunction_ReLU(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_SOFTRELU
    {
        // y = log(1 + exp(x))
        ActivationFunction_BNLL(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ABS
    {
        ActivationFunction_Abs(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_POWER
    {
        // y = (alpha + beta * x ) ^ gamma
        ActivationFunction_Power(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU
    {
        ActivationFunction_Clipped_ReLU(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LEAKY_RELU
    {
        ActivationFunction_Leaky_ReLU(n, res, data, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ELU
    {
        ActivationFunction_ELU(n, res, data, gamma, beta, alpha);
    }
#endif
}

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

__attribute__((always_inline)) void BatchNormFunctionSpatial(const uint n,
                                                             _FLOAT_PREC* out,
                                                             const _FLOAT* in,
                                                             const _FLOAT_PREC mean,
                                                             const _FLOAT_PREC invVariance,
                                                             const _FLOAT_PREC scale,
                                                             const _FLOAT_PREC bias)
{
    for(uint i = 0; i < n; ++i)
    {
        out[i] = mad(scale, ((_FLOAT_PREC)in[i] - mean) * invVariance, bias);
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

    int n_i = 0;
    __attribute__((opencl_unroll_hint(2))) for(n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        int index = n_i * MIO_BN_CHW + c_offset + hw_i * MIOPEN_READ_UNIT;

        *((MIOPEN_READ_TYPE*)data) = *((const __global MIOPEN_READ_TYPE*)(in + index));
        _FLOAT_PREC bnRes[MIOPEN_READ_UNIT];
        _FLOAT_PREC actRes[MIOPEN_READ_UNIT];
        BatchNormFunctionSpatial(MIOPEN_READ_UNIT, bnRes, data, pmean, invVariance, pscale, pbias);
        ActivationFunction(MIOPEN_READ_UNIT, actRes, bnRes, gamma, beta, alpha);
        for(int i = 0; i < MIOPEN_READ_UNIT; i++)
        {
            out[index + i] = (_FLOAT)actRes[i];
        }
    }
} // end spatial norm

__attribute__((always_inline)) void BatchNormFunctionPerAct(const uint n,
                                                            _FLOAT_PREC* out,
                                                            const _FLOAT* in,
                                                            const _FLOAT_PREC* mean,
                                                            const _FLOAT_PREC* invVariance,
                                                            const _FLOAT_PREC* scale,
                                                            const _FLOAT_PREC* bias)
{
    for(uint i = 0; i < n; ++i)
    {
        out[i] = mad(scale[i], ((_FLOAT_PREC)in[i] - mean[i]) * invVariance[i], bias[i]);
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

    for(int i          = 0; i < MIOPEN_READ_UNIT; i++)
        invVariance[i] = rsqrt((_FLOAT_PREC)pvar[i] + epsilon);

    int n_i = 0;
    __attribute__((opencl_unroll_hint(2))) for(n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        int index                  = n_i * MIO_BN_CHW + chw_i;
        *((MIOPEN_READ_TYPE*)data) = *((const __global MIOPEN_READ_TYPE*)(in + index));
        _FLOAT_PREC bnRes[MIOPEN_READ_UNIT];
        _FLOAT_PREC actRes[MIOPEN_READ_UNIT];
        BatchNormFunctionPerAct(MIOPEN_READ_UNIT, bnRes, data, pmean, invVariance, pscale, pbias);
        ActivationFunction(MIOPEN_READ_UNIT, actRes, bnRes, gamma, beta, alpha);
        for(int i = 0; i < MIOPEN_READ_UNIT; i++)
        {
            out[index + i] = (_FLOAT)actRes[i];
        }
    }
}

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif // #ifdef LITE
