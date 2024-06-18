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

#include "float_types.h"
#include "tensor_view.hpp"

__device__ FLOAT_ACCUM sigmoid(FLOAT_ACCUM x) { return 1 / (1 + exp(-x)); }
__device__ FLOAT_ACCUM calc_loss(FLOAT_ACCUM x, FLOAT_ACCUM y)
{
    FLOAT_ACCUM sig = sigmoid(x);
    return y * log(sig) + (1 - y) * log(1 - sig);
}
__device__ FLOAT_ACCUM calc_loss_grad(FLOAT_ACCUM x, FLOAT_ACCUM y)
{
    FLOAT_ACCUM sig = sigmoid(x);
    return y * (1 - sig) + (y - 1) * sig;
}

template <typename DTYPE>
__device__ void multilabelsoftmarginlossunreducedforward2d(const DTYPE* __restrict__ I,
                                                           const DTYPE* __restrict__ T,
                                                           const DTYPE* __restrict__ W,
                                                           DTYPE* __restrict__ O,
                                                           tensor_view_t<2> I_tv,
                                                           tensor_view_t<2> T_tv,
                                                           tensor_view_t<1> W_tv,
                                                           tensor_view_t<1> O_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = I_tv.size[0], C = I_tv.size[1];
    size_t n = gid;
    if(n >= N)
        return;

    FLOAT_ACCUM loss = 0;

    for(size_t c = 0; c < C; c++)
    {
        FLOAT_ACCUM w = CVT_FLOAT2ACCUM(W[W_tv.get_tensor_view_idx({c})]);
        FLOAT_ACCUM i = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, c})]);
        FLOAT_ACCUM t = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx({n, c})]);

        loss += -w * calc_loss(i, t);
    }

    O[O_tv.get_tensor_view_idx({n})] = CVT_ACCUM2FLOAT(loss / C);
}

extern "C" __global__ void
MultilabelSoftMarginLossUnreducedForward2d(const INPUT_TYPE* __restrict__ I,
                                           const INPUT_TYPE* __restrict__ T,
                                           const INPUT_TYPE* __restrict__ W,
                                           INPUT_TYPE* __restrict__ O,
                                           tensor_view_t<2> I_tv,
                                           tensor_view_t<2> T_tv,
                                           tensor_view_t<1> W_tv,
                                           tensor_view_t<1> O_tv)
{
    // instantiate the kernel
    multilabelsoftmarginlossunreducedforward2d<INPUT_TYPE>(I, T, W, O, I_tv, T_tv, W_tv, O_tv);
}

// template <typename DTYPE>
// __device__ void multilabelsoftmarginlossunreducedbackward2d(const DTYPE* __restrict__ I,
//                                                             const DTYPE* __restrict__ T,
//                                                             const DTYPE* __restrict__ W,
//                                                             const DTYPE* __restrict__ dO,
//                                                             DTYPE* __restrict__ dI,
//                                                             DTYPE* __restrict__ dW,
//                                                             tensor_view_t<2> I_tv,
//                                                             tensor_view_t<2> T_tv,
//                                                             tensor_view_t<1> W_tv,
//                                                             tensor_view_t<1> dO_tv,
//                                                             tensor_view_t<2> dI_tv,
//                                                             tensor_view_t<1> dW_tv)
// {
//     const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

//     size_t N = I_tv.size[0], C = I_tv.size[1];
//     size_t n = gid;

//     if(n >= N)
//         return;

//     for(size_t c = 0; c < C; c++)
//     {
//         FLOAT_ACCUM w = CVT_FLOAT2ACCUM(W[W_tv.get_tensor_view_idx({c})]);
//         FLOAT_ACCUM i = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, c})]);
//         FLOAT_ACCUM t = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx({n, c})]);
//         FLOAT_ACCUM o = CVT_FLOAT2ACCUM(dO[dO_tv.get_tensor_view_idx({n})]);

//         FLOAT_ACCUM grad                      = calc_loss_grad(i, t);
//         dI[dI_tv.get_tensor_view_idx({n, c})] = CVT_ACCUM2FLOAT(-w * o * grad / C);
//         atomicAdd(&dW[dW_tv.get_tensor_view_idx({c})], -o / C * calc_loss(i, t));
//     }
// }

// extern "C" __global__ void
// MultilabelSoftMarginLossUnreducedBackward2d(const INPUT_TYPE* __restrict__ I,
//                                             const INPUT_TYPE* __restrict__ T,
//                                             const INPUT_TYPE* __restrict__ W,
//                                             const INPUT_TYPE* __restrict__ dO,
//                                             INPUT_TYPE* __restrict__ dI,
//                                             INPUT_TYPE* __restrict__ dW,
//                                             tensor_view_t<2> I_tv,
//                                             tensor_view_t<2> T_tv,
//                                             tensor_view_t<1> W_tv,
//                                             tensor_view_t<1> dO_tv,
//                                             tensor_view_t<2> dI_tv,
//                                             tensor_view_t<1> dW_tv)
// {
//     // instantiate the kernel
//     multilabelsoftmarginlossunreducedbackward2d<INPUT_TYPE>(
//         I, T, W, dO, dI, dW, I_tv, T_tv, W_tv, dO_tv, dI_tv, dW_tv);
// }

template <typename DTYPE>
__device__ void multilabelsoftmarginlossforward2d(const DTYPE* __restrict__ I,
                                                  const DTYPE* __restrict__ T,
                                                  const DTYPE* __restrict__ W,
                                                  DTYPE* __restrict__ lsum,
                                                  const float divisor,
                                                  tensor_view_t<2> I_tv,
                                                  tensor_view_t<2> T_tv,
                                                  tensor_view_t<1> W_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t N = I_tv.size[0], C = I_tv.size[1];
    size_t n = gid;

    if(n >= N)
        return;

    FLOAT_ACCUM loss = 0;

    for(size_t c = 0; c < C; c++)
    {
        FLOAT_ACCUM w = CVT_FLOAT2ACCUM(W[W_tv.get_tensor_view_idx({c})]);
        FLOAT_ACCUM i = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, c})]);
        FLOAT_ACCUM t = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx({n, c})]);

        loss += -w * calc_loss(i, t);
    }

    lsum[n] = CVT_ACCUM2FLOAT(loss / C / divisor);
}

extern "C" __global__ void MultilabelSoftMarginLossForward2d(const INPUT_TYPE* __restrict__ I,
                                                             const INPUT_TYPE* __restrict__ T,
                                                             const INPUT_TYPE* __restrict__ W,
                                                             INPUT_TYPE* __restrict__ lsum,
                                                             const float divisor,
                                                             tensor_view_t<2> I_tv,
                                                             tensor_view_t<2> T_tv,
                                                             tensor_view_t<1> W_tv)
{
    // instantiate the kernel
    multilabelsoftmarginlossforward2d<INPUT_TYPE>(I, T, W, lsum, divisor, I_tv, T_tv, W_tv);
}

// template <typename DTYPE>
// __device__ void multilabelsoftmarginlossbackward2d(const DTYPE* __restrict__ I,
//                                                    const DTYPE* __restrict__ T,
//                                                    const DTYPE* __restrict__ W,
//                                                    const DTYPE* __restrict__ dO,
//                                                    DTYPE* __restrict__ dI,
//                                                    DTYPE* __restrict__ dW,
//                                                    const float divisor,
//                                                    tensor_view_t<2> I_tv,
//                                                    tensor_view_t<2> T_tv,
//                                                    tensor_view_t<1> W_tv,
//                                                    tensor_view_t<1> dO_tv,
//                                                    tensor_view_t<2> dI_tv,
//                                                    tensor_view_t<1> dW_tv)
// {
//     const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

//     size_t N = I_tv.size[0], C = I_tv.size[1];
//     size_t n = gid;

//     if(n >= N)
//         return;

//     for(size_t c = 0; c < C; c++)
//     {
//         FLOAT_ACCUM w = CVT_FLOAT2ACCUM(W[W_tv.get_tensor_view_idx({c})]);
//         FLOAT_ACCUM i = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx({n, c})]);
//         FLOAT_ACCUM t = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx({n, c})]);
//         FLOAT_ACCUM o = CVT_FLOAT2ACCUM(dO[dO_tv.get_tensor_view_idx({0})]);

//         FLOAT_ACCUM grad                      = calc_loss_grad(i, t);
//         dI[dI_tv.get_tensor_view_idx({n, c})] = CVT_ACCUM2FLOAT(-w * o * grad / C / divisor);
//         atomicAdd(&dW[dW_tv.get_tensor_view_idx({c})], -o / C * calc_loss(i, t) / divisor);
//     }
// }

// extern "C" __global__ void MultilabelSoftMarginLossBackward2d(const INPUT_TYPE* __restrict__ I,
//                                                               const INPUT_TYPE* __restrict__ T,
//                                                               const INPUT_TYPE* __restrict__ W,
//                                                               const INPUT_TYPE* __restrict__ dO,
//                                                               INPUT_TYPE* __restrict__ dI,
//                                                               INPUT_TYPE* __restrict__ dW,
//                                                               const float divisor,
//                                                               tensor_view_t<2> I_tv,
//                                                               tensor_view_t<2> T_tv,
//                                                               tensor_view_t<1> W_tv,
//                                                               tensor_view_t<1> dO_tv,
//                                                               tensor_view_t<2> dI_tv,
//                                                               tensor_view_t<1> dW_tv)
// {
//     // instantiate the kernel
//     multilabelsoftmarginlossbackward2d<INPUT_TYPE>(
//         I, T, W, dO, dI, dW, divisor, I_tv, T_tv, W_tv, dO_tv, dI_tv, dW_tv);
// }
