/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#define EPSILON (_FLOAT_PREC)0.0001
#elif(MIOPEN_USE_FP32 == 1 && MIOPEN_USE_FPMIX == 0)
#define _FLOAT float
#define _FLOAT_PREC float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#define EPSILON (_FLOAT)0.000001
#elif MIOPEN_USE_FPMIX == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_PREC float
#define EPSILON (_FLOAT)0.000001
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

#define UNUSED __attribute__((__unused__))

#define MIOPEN_NRN_GROUP_SZ2 1

#define MIOPEN_NEURON_PASTHRU 0      // x
#define MIOPEN_NEURON_LOGISTIC 1     // 1 / (1 + e^-x)  //Sigmoid
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

static __constant _FLOAT kBNLL_THRESHOLD = (_FLOAT)50.;

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
        res[i] = data[i] * (data[i] > 0);
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

//==================== PER ACTIVATION =======================

__kernel void MIOpenBatchNormActivFwdTrainPerActivation(
    const _FLOAT alpha,
    const _FLOAT beta,
    const _FLOAT gamma,
    double epsilon, /* input fuzz param > 0 */
#if(MIO_RUNNING_RESULT == 1)
    double expAvgFactor,
#endif
    const __global _FLOAT* __restrict in,        /* x input */
    __global _FLOAT* __restrict out,             /* y output */
    const __global _FLOAT_PREC* __restrict bias, /* beta 1xCxHxW */
    const __global _FLOAT_PREC* __restrict scale /* gamma 1xCxHxW */

#if(MIO_RUNNING_RESULT == 1)
    ,
    __global _FLOAT_PREC* __restrict runningMean,    /*input and output, same descriptor as bias*/
    __global _FLOAT_PREC* __restrict runningVariance /*input and output*/
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
    ,
    __global _FLOAT_PREC* __restrict savedInvVariance, /*output only*/
    __global _FLOAT_PREC* __restrict savedMean         /*output only*/

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
    _FLOAT_PREC bn_out, act_out;
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
    for(unsigned int img_offset = 0; img_offset < MIO_BN_HW; img_offset += yglb_sz)
    {
        inImgIndex = img_offset + ygid;
        if(inImgIndex < MIO_BN_HW)
        {
            mean     = (_FLOAT_PREC)0.;
            variance = (_FLOAT_PREC)0.;
            adjIndex = Cidx + inImgIndex; // gamma and beta tensor index

#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            {
                index           = MIO_BN_CHW * n + adjIndex;
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
            savedInvVariance[adjIndex] = invVariance; /*output only*/
            savedMean[adjIndex]        = mean;
#endif
#if(MIO_RUNNING_RESULT == 1)
            pvt_runMean    = *(runningMean + adjIndex); // previous: oldRunMean
            pvt_newRunMean = mad((_FLOAT_PREC)-expAvgFactor,
                                 pvt_runMean,
                                 pvt_runMean); // tmp = oldRunMean*(1-factor)

            runningMean[adjIndex] = mad((_FLOAT_PREC)mean,
                                        (_FLOAT_PREC)expAvgFactor,
                                        pvt_newRunMean); // newMean*factor + tmp

            const _FLOAT_PREC adjust = (MIO_BN_N == 1) ? variance : variance * (N / (N - 1));
            runningVariance[adjIndex] =
                (1 - (_FLOAT_PREC)expAvgFactor) * *(runningVariance + adjIndex) +
                (_FLOAT_PREC)expAvgFactor * adjust;
#endif

#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; n++)
            { // per (x-dims) channel load a block of data unsigned into LDS
                index  = MIO_BN_CHW * n + adjIndex;
                inhat  = ((_FLOAT_PREC)(*(in + index)) - mean) * invVariance;
                bn_out = mad(pvt_scale, inhat, pvt_bias);
                ActivationFunction(1, &act_out, &bn_out, gamma, beta, alpha);
                out[index] = (_FLOAT)act_out;
            } // end for(n)
        }     // end if(inImgIndex)
    }         // end for(img_offset) //image mini_batch is processed
}

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
