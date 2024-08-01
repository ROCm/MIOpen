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
#ifndef MIOPEN_NRN_OP_ID
#define MIOPEN_NRN_OP_ID 0
#endif

#if MIOPEN_USE_FP16 == 1
#define FP_TYPE half
#define FP_TYPE_PREC float
#define EPSILON static_cast<FP_TYPE>(0.0001)
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define FP_TYPE float
#define FP_TYPE_PREC float
#define EPSILON static_cast<FP_TYPE>(0.000001)
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#define FP_TYPE2 PPCAT(FP_TYPE, TWO)
#define FP_TYPE4 PPCAT(FP_TYPE, FOUR)
#define FP_TYPE8 PPCAT(FP_TYPE, EIGHT)
#define FP_TYPE_PREC2 PPCAT(FP_TYPE_PREC, TWO)
#define FP_TYPE_PREC4 PPCAT(FP_TYPE_PREC, FOUR)
#define FP_TYPE_PREC8 PPCAT(FP_TYPE_PREC, EIGHT)

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
#define MIOPEN_NEURON_TOTAL 10

#define kBNLL_THRESHOLD static_cast<FP_TYPE>(50.0)

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_PassThru(T (&__restrict__ res)[N],
                                                            const T (&__restrict__ data)[N],
                                                            const T /*gamma*/,
                                                            const T /*beta*/,
                                                            const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        res[i] = data[i];
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_ReLU(T (&__restrict__ res)[N],
                                                        const T (&__restrict__ data)[N],
                                                        const T /*gamma*/,
                                                        const T /*beta*/,
                                                        const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        res[i] = data[i] * (data[i] > 0);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Sigmoid(T (&__restrict__ res)[N],
                                                           const T (&__restrict__ data)[N],
                                                           const T /*gamma*/,
                                                           const T /*beta*/,
                                                           const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        // y = 1/(1 + exp(-x))
        res[i] = static_cast<T>(1) / (static_cast<T>(1) + exp(-data[i]));
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_TanH(T (&__restrict__ res)[N],
                                                        const T (&__restrict__ data)[N],
                                                        const T /*gamma*/,
                                                        const T beta,
                                                        const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        // y = beta * tanh(alpha * x)
        res[i] = beta * tanh(alpha * data[i]);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Abs(T (&__restrict__ res)[N],
                                                       const T (&__restrict__ data)[N],
                                                       const T /*gamma*/,
                                                       const T /*beta*/,
                                                       const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        res[i] = fabs(data[i]);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Square(T (&__restrict__ res)[N],
                                                          const T (&__restrict__ data)[N],
                                                          const T /*gamma*/,
                                                          const T /*beta*/,
                                                          const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        res[i] = data[i] * data[i];
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Sqrt(T (&__restrict__ res)[N],
                                                        const T (&__restrict__ data)[N],
                                                        const T /*gamma*/,
                                                        const T /*beta*/,
                                                        const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        res[i] = sqrt(data[i]);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Linear(T (&__restrict__ res)[N],
                                                          const T (&__restrict__ data)[N],
                                                          const T /*gamma*/,
                                                          const T beta,
                                                          const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        res[i] = alpha + beta * data[i];
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Power(T (&__restrict__ res)[N],
                                                         const T (&__restrict__ data)[N],
                                                         const T gamma,
                                                         const T beta,
                                                         const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        // y = (alpha + beta * x ) ^ gamma
        T arg  = alpha + data[i] * beta;
        res[i] = arg <= static_cast<T>(EPSILON) ? static_cast<T>(0) : pow(arg, gamma);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_BNLL(T (&__restrict__ res)[N],
                                                        const T (&__restrict__ data)[N],
                                                        const T /*gamma*/,
                                                        const T /*beta*/,
                                                        const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        //	y = log(1 + exp(x))
        res[i] = (data[i] > 0) ? (data[i] + log(static_cast<T>(1) + exp(-data[i])))
                               : log(static_cast<T>(1) + exp(data[i]));
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Leaky_ReLU(T (&__restrict__ res)[N],
                                                              const T (&__restrict__ data)[N],
                                                              const T /*gamma*/,
                                                              const T /*beta*/,
                                                              const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        res[i] = data[i] * ((data[i] > 0) ? static_cast<T>(1) : alpha);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Clipped_ReLU(T (&__restrict__ res)[N],
                                                                const T (&__restrict__ data)[N],
                                                                const T /*gamma*/,
                                                                const T /*beta*/,
                                                                const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        res[i] = fmin(static_cast<T>(alpha), fmax(static_cast<T>(data[i]), 0));
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_ELU(T (&__restrict__ res)[N],
                                                       const T (&__restrict__ data)[N],
                                                       const T /*gamma*/,
                                                       const T /*beta*/,
                                                       const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        res[i] = (data[i] > 0) ? data[i] : (alpha * (exp(data[i]) - static_cast<T>(1)));
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction(T (&__restrict__ res)[N],
                                                   const T (&__restrict__ data)[N],
                                                   const T gamma,
                                                   const T beta,
                                                   const T alpha)
{
    static_assert(N <= 4);
    if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU)
    {
        ActivationFunction_PassThru(res, data, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LOGISTIC)
    {
        ActivationFunction_Sigmoid(res, data, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_TANH)
    {
        ActivationFunction_TanH(res, data, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU)
    {
        ActivationFunction_ReLU(res, data, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_SOFTRELU)
    {
        ActivationFunction_BNLL(res, data, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ABS)
    {
        ActivationFunction_Abs(res, data, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_POWER)
    {
        ActivationFunction_Power(res, data, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU)
    {
        ActivationFunction_Clipped_ReLU(res, data, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LEAKY_RELU)
    {
        ActivationFunction_Leaky_ReLU(res, data, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ELU)
    {
        ActivationFunction_ELU(res, data, gamma, beta, alpha);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void
ActivationFunction_PassThru_Diff(T (&__restrict__ bot_diff)[N],
                                 const T (&__restrict__ top_diff)[N],
                                 const T* /*bot_data*/,
                                 const T* /*top_data*/,
                                 const T /*diff_scale*/,
                                 const T /*gamma*/,
                                 const T /*beta*/,
                                 const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        bot_diff[i] = top_diff[i];
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_ReLU_Diff(T (&__restrict__ bot_diff)[N],
                                                             const T (&__restrict__ top_diff)[N],
                                                             const T (&__restrict__ bot_data)[N],
                                                             const T* /*top_data*/,
                                                             const T /*diff_scale*/,
                                                             const T /*gamma*/,
                                                             const T /*beta*/,
                                                             const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        bot_diff[i] = top_diff[i] * (bot_data[i] > 0);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_TanH_Diff(T (&__restrict__ bot_diff)[N],
                                                             const T (&__restrict__ top_diff)[N],
                                                             const T* /*bot_data*/,
                                                             const T (&__restrict__ top_data)[N],
                                                             const T /*diff_scale*/,
                                                             const T /*gamma*/,
                                                             const T beta,
                                                             const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        // dy/dx = alpha * (beta - y^2 / beta)
        T y         = top_data[i];
        bot_diff[i] = fabs(beta) <= static_cast<T>(EPSILON)
                          ? static_cast<T>(0)
                          : (top_diff[i] * alpha * (beta - y * y / beta));
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Sigmoid_Diff(T (&__restrict__ bot_diff)[N],
                                                                const T (&__restrict__ top_diff)[N],
                                                                const T* /*bot_data*/,
                                                                const T (&__restrict__ top_data)[N],
                                                                const T /*diff_scale*/,
                                                                const T /*gamma*/,
                                                                const T /*beta*/,
                                                                const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        // y = 1/(1 + exp(-x))
        T sigmoid_x = top_data[i];
        bot_diff[i] = top_diff[i] * sigmoid_x * (static_cast<T>(1) - sigmoid_x);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Abs_Diff(T (&__restrict__ bot_diff)[N],
                                                            const T (&__restrict__ top_diff)[N],
                                                            const T (&__restrict__ bot_data)[N],
                                                            const T* /*top_data*/,
                                                            const T /*diff_scale*/,
                                                            const T /*gamma*/,
                                                            const T /*beta*/,
                                                            const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0) ? 1 : -1);
    }
}

// Compute dy/dx = beta * gamma * (alpha + beta * x)^(gamma - 1)
//               = diff_scale * y / (alpha + beta * x)
template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Power_Diff(T (&__restrict__ bot_diff)[N],
                                                              const T* /*top_diff*/,
                                                              const T (&__restrict__ bot_data)[N],
                                                              const T (&__restrict__ top_data)[N],
                                                              const T diff_scale,
                                                              const T /*gamma*/,
                                                              const T beta,
                                                              const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        T arg = alpha + bot_data[i] * beta;
        bot_diff[i] =
            arg <= static_cast<T>(EPSILON) ? static_cast<T>(0) : (diff_scale * top_data[i] / arg);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_BNLL_Diff(T (&__restrict__ bot_diff)[N],
                                                             const T (&__restrict__ top_diff)[N],
                                                             const T (&__restrict__ bot_data)[N],
                                                             const T* /*top_data*/,
                                                             const T /*diff_scale*/,
                                                             const T /*gamma*/,
                                                             const T /*beta*/,
                                                             const T /*alpha*/)
{
    for(uint i = 0; i < N; ++i)
    {
        // y = (log(1 + exp(x)))
        // dy/dx = 1/ (1 + exp(-x))
        T expval    = exp(fmin(static_cast<T>(bot_data[i]), static_cast<T>(kBNLL_THRESHOLD)));
        bot_diff[i] = top_diff[i] * expval / (expval + static_cast<T>(1));
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void
ActivationFunction_Leaky_ReLU_Diff(T (&__restrict__ bot_diff)[N],
                                   const T (&__restrict__ top_diff)[N],
                                   const T (&__restrict__ bot_data)[N],
                                   const T* /*top_data*/,
                                   const T /*diff_scale*/,
                                   const T /*gamma*/,
                                   const T /*beta*/,
                                   const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0) ? static_cast<T>(1) : alpha);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void
ActivationFunction_Clipped_ReLU_Diff(T (&__restrict__ bot_diff)[N],
                                     const T (&__restrict__ top_diff)[N],
                                     const T (&__restrict__ bot_data)[N],
                                     const T* /*top_data*/,
                                     const T /*diff_scale*/,
                                     const T /*gamma*/,
                                     const T /*beta*/,
                                     const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0 && bot_data[i] <= alpha) ? static_cast<T>(1)
                                                                               : static_cast<T>(0));
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_ELU_Diff(T (&__restrict__ bot_diff)[N],
                                                            const T (&__restrict__ top_diff)[N],
                                                            const T (&__restrict__ bot_data)[N],
                                                            const T (&__restrict__ top_data)[N],
                                                            const T /*diff_scale*/,
                                                            const T /*gamma*/,
                                                            const T /*beta*/,
                                                            const T alpha)
{
    for(uint i = 0; i < N; ++i)
    {
        bot_diff[i] = top_diff[i] * ((bot_data[i] > 0) ? 1 : top_data[i] + alpha);
    }
}

template <typename T, size_t N>
__forceinline__ __device__ void ActivationFunction_Diff(T (&__restrict__ bot_diff)[N],
                                                        const T (&__restrict__ top_diff)[N],
                                                        const T (&__restrict__ bot_data)[N],
                                                        const T (&__restrict__ top_data)[N],
                                                        const T diff_scale,
                                                        const T gamma,
                                                        const T beta,
                                                        const T alpha)
{
    static_assert(N <= 4);
    if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_PASTHRU)
    {
        ActivationFunction_PassThru_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LOGISTIC)
    {
        ActivationFunction_Sigmoid_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_TANH)
    {
        ActivationFunction_TanH_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_RELU)
    {
        ActivationFunction_ReLU_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_SOFTRELU)
    {
        ActivationFunction_BNLL_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ABS)
    {
        ActivationFunction_Abs_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_POWER)
    {
        ActivationFunction_Power_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_CLIPPED_RELU)
    {
        ActivationFunction_Clipped_ReLU_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_LEAKY_RELU)
    {
        ActivationFunction_Leaky_ReLU_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
    else if constexpr(MIOPEN_NRN_OP_ID == MIOPEN_NEURON_ELU)
    {
        ActivationFunction_ELU_Diff(
            bot_diff, top_diff, bot_data, top_data, diff_scale, gamma, beta, alpha);
    }
}
