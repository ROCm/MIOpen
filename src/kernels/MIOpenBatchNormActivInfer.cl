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
#define EPSILON (_FLOAT)0.0001
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define EPSILON (_FLOAT)0.000001
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

static __constant _FLOAT kBNLL_THRESHOLD = (_FLOAT)50.;

__attribute__((always_inline)) void ActivationFunction_PassThru(const uint n,
                                                                _FLOAT* res,
                                                                const _FLOAT* data,
                                                                UNUSED const _FLOAT gamma,
                                                                UNUSED const _FLOAT beta,
                                                                UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = data[i];
    }
}

__attribute__((always_inline)) void ActivationFunction_ReLU(const uint n,
                                                            _FLOAT* res,
                                                            const _FLOAT* data,
                                                            UNUSED const _FLOAT gamma,
                                                            UNUSED const _FLOAT beta,
                                                            UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = data[i] * (data[i] > 0);
    }
}

__attribute__((always_inline)) void ActivationFunction_Sigmoid(const uint n,
                                                               _FLOAT* res,
                                                               const _FLOAT* data,
                                                               UNUSED const _FLOAT gamma,
                                                               UNUSED const _FLOAT beta,
                                                               UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = 1/(1 + exp(-x))
        res[i] = (_FLOAT)1.f / ((_FLOAT)1.f + exp(-data[i]));
    }
}

__attribute__((always_inline)) void ActivationFunction_TanH(const uint n,
                                                            _FLOAT* res,
                                                            const _FLOAT* data,
                                                            UNUSED const _FLOAT gamma,
                                                            const _FLOAT beta,
                                                            const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = beta * tanh(alpha * x)
        res[i] = beta * tanh(alpha * data[i]);
    }
}

__attribute__((always_inline)) void ActivationFunction_Abs(const uint n,
                                                           _FLOAT* res,
                                                           const _FLOAT* data,
                                                           UNUSED const _FLOAT gamma,
                                                           UNUSED const _FLOAT beta,
                                                           UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = fabs(data[i]);
    }
}

__attribute__((always_inline)) void ActivationFunction_Square(const uint n,
                                                              _FLOAT* res,
                                                              const _FLOAT* data,
                                                              UNUSED const _FLOAT gamma,
                                                              UNUSED const _FLOAT beta,
                                                              UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {

        res[i] = data[i] * data[i];
    }
}

__attribute__((always_inline)) void ActivationFunction_Sqrt(const uint n,
                                                            _FLOAT* res,
                                                            const _FLOAT* data,
                                                            UNUSED const _FLOAT gamma,
                                                            UNUSED const _FLOAT beta,
                                                            UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {

        res[i] = sqrt(data[i]);
    }
}

__attribute__((always_inline)) void ActivationFunction_Linear(const uint n,
                                                              _FLOAT* res,
                                                              const _FLOAT* data,
                                                              UNUSED const _FLOAT gamma,
                                                              const _FLOAT beta,
                                                              const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = alpha + beta * data[i];
    }
}

__attribute__((always_inline)) void ActivationFunction_Power(const uint n,
                                                             _FLOAT* res,
                                                             const _FLOAT* data,
                                                             const _FLOAT gamma,
                                                             const _FLOAT beta,
                                                             const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = (alpha + beta * x ) ^ gamma
        _FLOAT arg = alpha + data[i] * beta;
        res[i]     = arg <= EPSILON ? (_FLOAT)0 : pow(arg, gamma);
    }
}

__attribute__((always_inline)) void ActivationFunction_BNLL(const uint n,
                                                            _FLOAT* res,
                                                            const _FLOAT* data,
                                                            UNUSED const _FLOAT gamma,
                                                            UNUSED const _FLOAT beta,
                                                            UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        //	y = log(1 + exp(x))
        res[i] = (data[i] > 0) ? (data[i] + log((_FLOAT)1.f + exp(-data[i])))
                               : log((_FLOAT)(1.f) + exp(data[i]));
    }
}

__attribute__((always_inline)) void ActivationFunction_Leaky_ReLU(const uint n,
                                                                  _FLOAT* res,
                                                                  const _FLOAT* data,
                                                                  UNUSED const _FLOAT gamma,
                                                                  UNUSED const _FLOAT beta,
                                                                  const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = data[i] * ((data[i] > 0) ? (_FLOAT)1.f : alpha);
    }
}

__attribute__((always_inline)) void ActivationFunction_Clipped_ReLU(const uint n,
                                                                    _FLOAT* res,
                                                                    const _FLOAT* data,
                                                                    UNUSED const _FLOAT gamma,
                                                                    UNUSED const _FLOAT beta,
                                                                    const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = fmin(alpha, fmax(data[i], 0));
    }
}

__attribute__((always_inline)) void ActivationFunction_ELU(const uint n,
                                                           _FLOAT* res,
                                                           const _FLOAT* data,
                                                           UNUSED const _FLOAT gamma,
                                                           UNUSED const _FLOAT beta,
                                                           const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        res[i] = (data[i] > 0) ? data[i] : (alpha * (exp(data[i]) - (_FLOAT)1.f));
    }
}

__attribute__((always_inline)) void ActivationFunction(const uint n,
                                                       _FLOAT* res,
                                                       const _FLOAT* data,
                                                       const _FLOAT gamma,
                                                       const _FLOAT beta,
                                                       const _FLOAT alpha)
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

/******************************************************************************/
/*                                  DIFF                                      */
/******************************************************************************/
__attribute__((always_inline)) void ActivationFunction_PassThru_Diff(const uint n,
                                                                     _FLOAT* bot_diff,
                                                                     const _FLOAT* top_diff,
                                                                     UNUSED const _FLOAT* bot_data,
                                                                     UNUSED const _FLOAT* top_data,
                                                                     UNUSED const _FLOAT diff_scale,
                                                                     UNUSED const _FLOAT gamma,
                                                                     UNUSED const _FLOAT beta,
                                                                     UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i];
    }
}

__attribute__((always_inline)) void ActivationFunction_ReLU_Diff(const uint n,
                                                                 _FLOAT* bot_diff,
                                                                 const _FLOAT* top_diff,
                                                                 const _FLOAT* bot_data,
                                                                 UNUSED const _FLOAT* top_data,
                                                                 UNUSED const _FLOAT diff_scale,
                                                                 UNUSED const _FLOAT gamma,
                                                                 UNUSED const _FLOAT beta,
                                                                 UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * (bot_data[i] > 0);
    }
}

__attribute__((always_inline)) void ActivationFunction_TanH_Diff(const uint n,
                                                                 _FLOAT* bot_diff,
                                                                 const _FLOAT* top_diff,
                                                                 UNUSED const _FLOAT* bot_data,
                                                                 const _FLOAT* top_data,
                                                                 UNUSED const _FLOAT diff_scale,
                                                                 UNUSED const _FLOAT gamma,
                                                                 const _FLOAT beta,
                                                                 const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // dy/dx = alpha * (beta - y^2 / beta)
        _FLOAT y = top_data[i];
        bot_diff[i] =
            fabs(beta) <= EPSILON ? (_FLOAT)0 : (top_diff[i] * alpha * (beta - y * y / beta));
    }
}

__attribute__((always_inline)) void ActivationFunction_Sigmoid_Diff(const uint n,
                                                                    _FLOAT* bot_diff,
                                                                    const _FLOAT* top_diff,
                                                                    UNUSED const _FLOAT* bot_data,
                                                                    const _FLOAT* top_data,
                                                                    UNUSED const _FLOAT diff_scale,
                                                                    UNUSED const _FLOAT gamma,
                                                                    UNUSED const _FLOAT beta,
                                                                    UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = 1/(1 + exp(-x))
        _FLOAT sigmoid_x = top_data[i];
        bot_diff[i]      = top_diff[i] * sigmoid_x * ((_FLOAT)1.f - sigmoid_x);
    }
}

__attribute__((always_inline)) void ActivationFunction_Abs_Diff(const uint n,
                                                                _FLOAT* bot_diff,
                                                                const _FLOAT* top_diff,
                                                                const _FLOAT* bot_data,
                                                                UNUSED const _FLOAT* top_data,
                                                                UNUSED const _FLOAT diff_scale,
                                                                UNUSED const _FLOAT gamma,
                                                                UNUSED const _FLOAT beta,
                                                                UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0) ? 1 : -1);
    }
}

// Compute dy/dx = beta * gamma * (alpha + beta * x)^(gamma - 1)
//               = diff_scale * y / (alpha + beta * x)
__attribute__((always_inline)) void ActivationFunction_Power_Diff(const uint n,
                                                                  _FLOAT* bot_diff,
                                                                  UNUSED const _FLOAT* top_diff,
                                                                  const _FLOAT* bot_data,
                                                                  const _FLOAT* top_data,
                                                                  const _FLOAT diff_scale,
                                                                  UNUSED const _FLOAT gamma,
                                                                  const _FLOAT beta,
                                                                  const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        _FLOAT arg  = alpha + bot_data[i] * beta;
        bot_diff[i] = arg <= EPSILON ? (_FLOAT)0 : (diff_scale * top_data[i] / arg);
    }
}

__attribute__((always_inline)) void ActivationFunction_BNLL_Diff(const uint n,
                                                                 _FLOAT* bot_diff,
                                                                 const _FLOAT* top_diff,
                                                                 const _FLOAT* bot_data,
                                                                 UNUSED const _FLOAT* top_data,
                                                                 UNUSED const _FLOAT diff_scale,
                                                                 UNUSED const _FLOAT gamma,
                                                                 UNUSED const _FLOAT beta,
                                                                 UNUSED const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        // y = (log(1 + exp(x)))
        // dy/dx = 1/ (1 + exp(-x))
        _FLOAT expval = exp(fmin(bot_data[i], kBNLL_THRESHOLD));
        bot_diff[i]   = top_diff[i] * expval / (expval + (_FLOAT)1.f);
    }
}

__attribute__((always_inline)) void
ActivationFunction_Leaky_ReLU_Diff(const uint n,
                                   _FLOAT* bot_diff,
                                   const _FLOAT* top_diff,
                                   const _FLOAT* bot_data,
                                   UNUSED const _FLOAT* top_data,
                                   UNUSED const _FLOAT diff_scale,
                                   UNUSED const _FLOAT gamma,
                                   UNUSED const _FLOAT beta,
                                   const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0) ? (_FLOAT)1.f : alpha);
    }
}

__attribute__((always_inline)) void
ActivationFunction_Clipped_ReLU_Diff(const uint n,
                                     _FLOAT* bot_diff,
                                     const _FLOAT* top_diff,
                                     const _FLOAT* bot_data,
                                     UNUSED const _FLOAT* top_data,
                                     UNUSED const _FLOAT diff_scale,
                                     UNUSED const _FLOAT gamma,
                                     UNUSED const _FLOAT beta,
                                     const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] =
            top_diff[i] * ((bot_data[i] > 0 && bot_data[i] <= alpha) ? (_FLOAT)1.f : (_FLOAT)0.f);
    }
}

__attribute__((always_inline)) void ActivationFunction_ELU_Diff(const uint n,
                                                                _FLOAT* bot_diff,
                                                                const _FLOAT* top_diff,
                                                                const _FLOAT* bot_data,
                                                                const _FLOAT* top_data,
                                                                UNUSED const _FLOAT diff_scale,
                                                                UNUSED const _FLOAT gamma,
                                                                UNUSED const _FLOAT beta,
                                                                const _FLOAT alpha)
{
    for(uint i = 0; i < n; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0) ? 1 : top_data[i] + alpha);
    }
}

__attribute__((always_inline)) void ActivationFunction_Diff(const uint n,
                                                            _FLOAT* bot_diff,
                                                            const _FLOAT* top_diff,
                                                            const _FLOAT* bot_data,
                                                            const _FLOAT* top_data,
                                                            const _FLOAT diff_scale,
                                                            const _FLOAT gamma,
                                                            const _FLOAT beta,
                                                            const _FLOAT alpha)
{

#if MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU
    {
        ActivationFunction_PassThru_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LOGISTIC
    {
        // y = 1/(1 + exp(-x))
        ActivationFunction_Sigmoid_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_TANH
    {
        // y = beta * tanh(alpha * x)
        ActivationFunction_TanH_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU
    {
        ActivationFunction_ReLU_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_SOFTRELU
    {
        // y = log(1 + exp(x))
        ActivationFunction_BNLL_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ABS
    {
        ActivationFunction_Abs_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_POWER
    {
        // y = (alpha + beta * x ) ^ gamma
        ActivationFunction_Power_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU
    {
        ActivationFunction_Clipped_ReLU_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LEAKY_RELU
    {
        ActivationFunction_Leaky_ReLU_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
#elif MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ELU
    {
        ActivationFunction_ELU_Diff(
            n, bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
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
                                                             _FLOAT* out,
                                                             const _FLOAT* in,
                                                             const _FLOAT mean,
                                                             const _FLOAT invVariance,
                                                             const _FLOAT scale,
                                                             const _FLOAT bias)
{
    for(uint i = 0; i < n; ++i)
    {
        out[i] = mad(scale, (in[i] - mean) * invVariance, bias);
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormActivInferSpatialEst(const _FLOAT alpha,
                                    const _FLOAT beta,
                                    const _FLOAT gamma,
                                    double epsilon,
                                    const __global _FLOAT* __restrict in, /* x input */
                                    __global _FLOAT* __restrict out,      /* y output */
                                    const __global _FLOAT* __restrict bias,
                                    const __global _FLOAT* __restrict scale,
                                    const __global _FLOAT* __restrict estimatedMean,
                                    const __global _FLOAT* __restrict estimatedVariance)
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);

    __local _FLOAT lmean;
    __local _FLOAT lvar;
    __local _FLOAT lscale;
    __local _FLOAT lbias;

    int c_i = gid1;
    // int n_i  = iDiv(gid0, MIO_BN_HW_RD);
    // int hw_i = iMod(gid0, n_i, MIO_BN_HW_RD);
    int hw_i = gid0;

    unsigned int c_offset = c_i * MIO_BN_HW;

    if(get_local_id(0) == 0)
    {
        lmean  = *(estimatedMean + c_i);
        lvar   = *(estimatedVariance + c_i);
        lscale = *(scale + c_i); // dims 1xCx1x1
        lbias  = *(bias + c_i);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    _FLOAT pmean  = lmean;
    _FLOAT pvar   = lvar;
    _FLOAT pscale = lscale;
    _FLOAT pbias  = lbias;

    _FLOAT data[MIOPEN_READ_UNIT];
    _FLOAT response[MIOPEN_READ_UNIT];
    _FLOAT invVariance = rsqrt(fabs(pvar + epsilon));

    int n_i = 0;
    __attribute__((opencl_unroll_hint(2))) for(n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        int index                  = n_i * MIO_BN_CHW + c_offset + hw_i * MIOPEN_READ_UNIT;
        *((MIOPEN_READ_TYPE*)data) = *((const __global MIOPEN_READ_TYPE*)(in + index));
        BatchNormFunctionSpatial(
            MIOPEN_READ_UNIT, response, (const _FLOAT*)data, pmean, invVariance, pscale, pbias);
        ActivationFunction(MIOPEN_READ_UNIT, data, (const _FLOAT*)response, gamma, beta, alpha);
        *((__global MIOPEN_READ_TYPE*)(out + index)) = *((MIOPEN_READ_TYPE*)data);
    }
} // end spatial norm

__attribute__((always_inline)) void BatchNormFunctionPerAct(const uint n,
                                                            _FLOAT* out,
                                                            const _FLOAT* in,
                                                            const _FLOAT* mean,
                                                            const _FLOAT* invVariance,
                                                            const _FLOAT* scale,
                                                            const _FLOAT* bias)
{
    for(uint i = 0; i < n; ++i)
    {
        out[i] = mad(scale[i], (in[i] - mean[i]) * invVariance[i], bias[i]);
    }
}

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormActivInferPerActEst(
    const _FLOAT alpha,
    const _FLOAT beta,
    const _FLOAT gamma,
    double epsilon,
    const __global _FLOAT* in,                       /* x input */
    __global _FLOAT* __restrict out,                 /* y output */
    const __global _FLOAT* __restrict bias,          /* beta 1xCxHxW */
    const __global _FLOAT* __restrict scale,         /* gamma 1xCxHxW */
    const __global _FLOAT* __restrict estimatedMean, /*input and output, same descriptor as bias*/
    const __global _FLOAT* __restrict estimatedVariance /*input and output*/)
{
    int gid0 = get_global_id(0);
    // int gid1 = get_global_id(1);

    int chw_i = gid0 * MIOPEN_READ_UNIT;

    _FLOAT pmean[MIOPEN_READ_UNIT];
    _FLOAT pvar[MIOPEN_READ_UNIT];
    _FLOAT pscale[MIOPEN_READ_UNIT];
    _FLOAT pbias[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)pmean)  = *((const __global MIOPEN_READ_TYPE*)(estimatedMean + chw_i));
    *((MIOPEN_READ_TYPE*)pvar)   = *((const __global MIOPEN_READ_TYPE*)(estimatedVariance + chw_i));
    *((MIOPEN_READ_TYPE*)pscale) = *((const __global MIOPEN_READ_TYPE*)(scale + chw_i));
    *((MIOPEN_READ_TYPE*)pbias)  = *((const __global MIOPEN_READ_TYPE*)(bias + chw_i));

    //_FLOAT pmean   = estimatedMean[chw_i];
    //_FLOAT pvar    = estimatedVariance[chw_i];
    //_FLOAT pscale   = scale[chw_i];
    //_FLOAT pbias    = bias[chw_i];

    _FLOAT data[MIOPEN_READ_UNIT];
    _FLOAT response[MIOPEN_READ_UNIT];
    _FLOAT pinvVariance[MIOPEN_READ_UNIT];

    for(int i           = 0; i < MIOPEN_READ_UNIT; i++)
        pinvVariance[i] = rsqrt(fabs(pvar[i] + epsilon));

    int n_i = 0;
    __attribute__((opencl_unroll_hint(2))) for(n_i = 0; n_i < MIO_BN_N; n_i++)
    {
        int index                  = n_i * MIO_BN_CHW + chw_i;
        *((MIOPEN_READ_TYPE*)data) = *((const __global MIOPEN_READ_TYPE*)(in + index));
        BatchNormFunctionPerAct(
            MIOPEN_READ_UNIT, response, (const _FLOAT*)data, pmean, pinvVariance, pscale, pbias);
        ActivationFunction(MIOPEN_READ_UNIT, data, (const _FLOAT*)response, gamma, beta, alpha);
        *((__global MIOPEN_READ_TYPE*)(out + index)) = *((MIOPEN_READ_TYPE*)data);
    }
}

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop

#endif // #ifdef LITE
