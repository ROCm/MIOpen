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

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE float
#endif

#ifndef D_TYPE
#define D_TYPE float
#endif

#ifndef REDUCE_SIZE
#define REDUCE_SIZE 256
#endif

template <typename TI, typename TO>
__device__ void nlllossForward5d(const TI* __restrict__ input,
                                 const int32_t* __restrict__ target,
                                 const TI* weight,
                                 TO* __restrict__ loss_sum,
                                 int32_t ignore_index,
                                 float divisor,
                                 tensor_view_5d_t input_tv,
                                 tensor_view_4d_t target_tv,
                                 tensor_view_1d_t weight_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[4];
    GET_NCDH(n[0], n[1], n[2], n[3], gid, target_tv);

    if(n[0] >= target_tv.size[0])
        return;

    size_t Tidx = TV4D_IDX(target_tv, n[0], n[1], n[2], n[3]);

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[Tidx];
    size_t Iidx = TV5D_IDX(input_tv, n[0], t, n[1], n[2], n[3]);
    size_t Widx = TV1D_IDX(weight_tv, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        loss_sum[gid] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[Widx]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[Iidx]);
    FLOAT_ACCUM d           = (divisor ? CVT_FP32_2ACCUM(divisor) : CVT_FP32_2ACCUM(1.0f));
    FLOAT_ACCUM val         = (CVT_FP32_2ACCUM(-1.0f) * w * input_value) / d;
    loss_sum[gid]           = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossForward5d(INPUT_TYPE* __restrict__ input,
                                            const int32_t* __restrict__ target,
                                            const INPUT_TYPE* weight,
                                            OUTPUT_TYPE* __restrict__ loss_sum,
                                            int32_t ignore_index,
                                            float divisor,
                                            tensor_view_5d_t input_tv,
                                            tensor_view_4d_t target_tv,
                                            tensor_view_1d_t weight_tv)
{
    nlllossForward5d<INPUT_TYPE, OUTPUT_TYPE>(
        input, target, weight, loss_sum, ignore_index, divisor, input_tv, target_tv, weight_tv);
}

template <typename TI, typename TO>
__device__ void nlllossBackward2d(TO* __restrict__ input_grad,
                                  const int32_t* __restrict__ target,
                                  const TI* weight,
                                  TI* __restrict__ output_grad,
                                  int32_t ignore_index,
                                  float divisor,
                                  tensor_view_2d_t input_grad_tv,
                                  tensor_view_1d_t target_tv,
                                  tensor_view_1d_t weight_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= target_tv.size[0])
        return;

    size_t Tidx = TV1D_IDX(target_tv, gid);

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[Tidx];
    size_t Iidx = TV2D_IDX(input_grad_tv, gid, t);
    size_t Widx = TV1D_IDX(weight_tv, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        input_grad[Iidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[Widx]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[0]);
    FLOAT_ACCUM d                = (divisor ? divisor : CVT_FP32_2ACCUM(1.0f));
    FLOAT_ACCUM input_grad_value = (CVT_FP32_2ACCUM(-1.0f) * w * grad_val) / d;

    input_grad[Iidx] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossBackward2d(INPUT_TYPE* __restrict__ input_grad,
                                             const int32_t* __restrict__ target,
                                             const INPUT_TYPE* weight,
                                             OUTPUT_TYPE* __restrict__ output_grad,
                                             int32_t ignore_index,
                                             float divisor,
                                             tensor_view_2d_t input_grad_tv,
                                             tensor_view_1d_t target_tv,
                                             tensor_view_1d_t weight_tv)
{
    nlllossBackward2d<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                               target,
                                               weight,
                                               output_grad,
                                               ignore_index,
                                               divisor,
                                               input_grad_tv,
                                               target_tv,
                                               weight_tv);
}

template <typename TI, typename TO>
__device__ void nlllossBackward5d(TO* __restrict__ input_grad,
                                  const int32_t* __restrict__ target,
                                  const TI* weight,
                                  TI* __restrict__ output_grad,
                                  int32_t ignore_index,
                                  float divisor,
                                  tensor_view_5d_t input_grad_tv,
                                  tensor_view_4d_t target_tv,
                                  tensor_view_1d_t weight_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[4];
    GET_NCDH(n[0], n[1], n[2], n[3], gid, target_tv);

    if(n[0] >= target_tv.size[0])
        return;

    size_t Tidx = TV4D_IDX(target_tv, n[0], n[1], n[2], n[3]);

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[Tidx];
    size_t Iidx = TV5D_IDX(input_grad_tv, n[0], t, n[1], n[2], n[3]);
    size_t Widx = TV1D_IDX(weight_tv, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        input_grad[Iidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[Widx]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[0]);
    FLOAT_ACCUM d                = (divisor ? divisor : CVT_FP32_2ACCUM(1.0f));
    FLOAT_ACCUM input_grad_value = (CVT_FP32_2ACCUM(-1.0f) * w * grad_val) / d;

    input_grad[Iidx] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossBackward5d(INPUT_TYPE* __restrict__ input_grad,
                                             const int32_t* __restrict__ target,
                                             const INPUT_TYPE* weight,
                                             OUTPUT_TYPE* __restrict__ output_grad,
                                             int32_t ignore_index,
                                             float divisor,
                                             tensor_view_5d_t input_grad_tv,
                                             tensor_view_4d_t target_tv,
                                             tensor_view_1d_t weight_tv)
{
    nlllossBackward5d<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                               target,
                                               weight,
                                               output_grad,
                                               ignore_index,
                                               divisor,
                                               input_grad_tv,
                                               target_tv,
                                               weight_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedForward5d(const TI* __restrict__ input,
                                          const int32_t* __restrict__ target,
                                          const TI* weight,
                                          TO* __restrict__ output,
                                          int32_t ignore_index,
                                          tensor_view_5d_t input_tv,
                                          tensor_view_4d_t target_tv,
                                          tensor_view_1d_t weight_tv,
                                          tensor_view_4d_t output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[4];
    GET_NCDH(n[0], n[1], n[2], n[3], gid, output_tv);

    if(n[0] >= output_tv.size[0])
        return;

    size_t Tidx = TV4D_IDX(target_tv, n[0], n[1], n[2], n[3]);
    size_t Oidx = TV4D_IDX(output_tv, n[0], n[1], n[2], n[3]);

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[Tidx];
    size_t Iidx = TV5D_IDX(input_tv, n[0], t, n[1], n[2], n[3]);
    size_t Widx = TV1D_IDX(weight_tv, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        output[Oidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[Widx]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[Iidx]);

    FLOAT_ACCUM val = CVT_FP32_2ACCUM(-1.0f) * w * input_value;
    output[Oidx]    = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward5d(const INPUT_TYPE* __restrict__ input,
                                                     const int32_t* __restrict__ target,
                                                     const INPUT_TYPE* weight,
                                                     OUTPUT_TYPE* __restrict__ output,
                                                     int32_t ignore_index,
                                                     tensor_view_5d_t input_tv,
                                                     tensor_view_4d_t target_tv,
                                                     tensor_view_1d_t weight_tv,
                                                     tensor_view_4d_t output_tv)
{
    nlllossUnreducedForward5d<INPUT_TYPE, OUTPUT_TYPE>(
        input, target, weight, output, ignore_index, input_tv, target_tv, weight_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedForward4d(const TI* __restrict__ input,
                                          const int32_t* __restrict__ target,
                                          const TI* weight,
                                          TO* __restrict__ output,
                                          int32_t ignore_index,
                                          tensor_view_4d_t input_tv,
                                          tensor_view_3d_t target_tv,
                                          tensor_view_1d_t weight_tv,
                                          tensor_view_3d_t output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[3];
    GET_NCD(n[0], n[1], n[2], gid, output_tv);

    if(n[0] >= output_tv.size[0])
        return;

    size_t Tidx = TV3D_IDX(target_tv, n[0], n[1], n[2]);
    size_t Oidx = TV3D_IDX(output_tv, n[0], n[1], n[2]);

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[Tidx];
    size_t Iidx = TV4D_IDX(input_tv, n[0], t, n[1], n[2]);
    size_t Widx = TV1D_IDX(weight_tv, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        output[Oidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[Widx]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[Iidx]);

    FLOAT_ACCUM val = CVT_FP32_2ACCUM(-1.0f) * w * input_value;
    output[Oidx]    = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward4d(const INPUT_TYPE* __restrict__ input,
                                                     const int32_t* __restrict__ target,
                                                     const INPUT_TYPE* weight,
                                                     OUTPUT_TYPE* __restrict__ output,
                                                     int32_t ignore_index,
                                                     tensor_view_4d_t input_tv,
                                                     tensor_view_3d_t target_tv,
                                                     tensor_view_1d_t weight_tv,
                                                     tensor_view_3d_t output_tv)
{
    nlllossUnreducedForward4d<INPUT_TYPE, OUTPUT_TYPE>(
        input, target, weight, output, ignore_index, input_tv, target_tv, weight_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedForward4dContiguous(const TI* __restrict__ input,
                                                    const int32_t* __restrict__ target,
                                                    const TI* weight,
                                                    TO* __restrict__ output,
                                                    int32_t ignore_index,
                                                    tensor_view_4d_t input_tv,
                                                    tensor_view_3d_t target_tv,
                                                    tensor_view_1d_t weight_tv,
                                                    tensor_view_3d_t output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[3];
    GET_NCD(n[0], n[1], n[2], gid, output_tv);

    if(n[0] >= output_tv.size[0])
        return;

    int32_t C   = input_tv.size[1];
    int32_t t   = target[gid];
    size_t Iidx = TV4D_IDX(input_tv, n[0], t, n[1], n[2]);

    if(t < 0 || t == ignore_index || t >= C)
    {
        output[gid] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[Iidx]);

    FLOAT_ACCUM val = CVT_FP32_2ACCUM(-1.0f) * w * input_value;
    output[gid]     = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward4dContiguous(const INPUT_TYPE* __restrict__ input,
                                                               const int32_t* __restrict__ target,
                                                               const INPUT_TYPE* weight,
                                                               OUTPUT_TYPE* __restrict__ output,
                                                               int32_t ignore_index,
                                                               tensor_view_4d_t input_tv,
                                                               tensor_view_3d_t target_tv,
                                                               tensor_view_1d_t weight_tv,
                                                               tensor_view_3d_t output_tv)
{
    nlllossUnreducedForward4dContiguous<INPUT_TYPE, OUTPUT_TYPE>(
        input, target, weight, output, ignore_index, input_tv, target_tv, weight_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedForward2d(const TI* __restrict__ input,
                                          const int32_t* __restrict__ target,
                                          const TI* weight,
                                          TO* __restrict__ output,
                                          int32_t ignore_index,
                                          tensor_view_2d_t input_tv,
                                          tensor_view_1d_t target_tv,
                                          tensor_view_1d_t weight_tv,
                                          tensor_view_1d_t output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= output_tv.size[0])
        return;

    size_t Tidx = TV1D_IDX(target_tv, gid);
    size_t Oidx = TV1D_IDX(output_tv, gid);

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[Tidx];
    size_t Iidx = TV2D_IDX(input_tv, gid, t);
    size_t Widx = TV1D_IDX(weight_tv, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        output[Oidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[Widx]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[Iidx]);

    FLOAT_ACCUM val = CVT_FP32_2ACCUM(-1.0f) * w * input_value;
    output[Oidx]    = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward2d(const INPUT_TYPE* __restrict__ input,
                                                     const int32_t* __restrict__ target,
                                                     const INPUT_TYPE* weight,
                                                     OUTPUT_TYPE* __restrict__ output,
                                                     int32_t ignore_index,
                                                     tensor_view_2d_t input_tv,
                                                     tensor_view_1d_t target_tv,
                                                     tensor_view_1d_t weight_tv,
                                                     tensor_view_1d_t output_tv)
{
    nlllossUnreducedForward2d<INPUT_TYPE, OUTPUT_TYPE>(
        input, target, weight, output, ignore_index, input_tv, target_tv, weight_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedForward2dContiguous(const TI* __restrict__ input,
                                                    const int32_t* __restrict__ target,
                                                    const TI* weight,
                                                    TO* __restrict__ output,
                                                    int32_t ignore_index,
                                                    tensor_view_2d_t input_tv,
                                                    tensor_view_1d_t target_tv,
                                                    tensor_view_1d_t weight_tv,
                                                    tensor_view_1d_t output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= output_tv.size[0])
        return;

    int32_t t = target[gid];
    size_t C  = weight_tv.size[0];
    if(t < 0 || t == ignore_index || t >= C)
    {
        output[gid] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);

    uint32_t input_offset   = gid * C + t;
    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[input_offset]);

    FLOAT_ACCUM val = CVT_FP32_2ACCUM(-1.0f) * w * input_value;
    output[gid]     = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward2dContiguous(const INPUT_TYPE* __restrict__ input,
                                                               const int32_t* __restrict__ target,
                                                               const INPUT_TYPE* weight,
                                                               OUTPUT_TYPE* __restrict__ output,
                                                               int32_t ignore_index,
                                                               tensor_view_2d_t input_tv,
                                                               tensor_view_1d_t target_tv,
                                                               tensor_view_1d_t weight_tv,
                                                               tensor_view_1d_t output_tv)
{
    nlllossUnreducedForward2dContiguous<INPUT_TYPE, OUTPUT_TYPE>(
        input, target, weight, output, ignore_index, input_tv, target_tv, weight_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedBackward5d(TO* __restrict__ input_grad,
                                           const int32_t* __restrict__ target,
                                           const TI* weight,
                                           TI* __restrict__ output_grad,
                                           int32_t ignore_index,
                                           tensor_view_5d_t input_grad_tv,
                                           tensor_view_4d_t target_tv,
                                           tensor_view_1d_t weight_tv,
                                           tensor_view_4d_t output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[4];
    GET_NCDH(n[0], n[1], n[2], n[3], gid, output_grad_tv);

    if(n[0] >= output_grad_tv.size[0])
        return;

    size_t Tidx = TV4D_IDX(target_tv, n[0], n[1], n[2], n[3]);
    size_t Oidx = TV4D_IDX(output_grad_tv, n[0], n[1], n[2], n[3]);

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[Tidx];
    size_t Iidx = TV5D_IDX(input_grad_tv, n[0], t, n[1], n[2], n[3]);
    size_t Widx = TV1D_IDX(weight_tv, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        input_grad[Iidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[Widx]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[Oidx]);
    FLOAT_ACCUM input_grad_value = CVT_FP32_2ACCUM(-1.0f) * w * grad_val;

    input_grad[Iidx] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossUnreducedBackward5d(INPUT_TYPE* __restrict__ input_grad,
                                                      const int32_t* __restrict__ target,
                                                      const INPUT_TYPE* weight,
                                                      OUTPUT_TYPE* __restrict__ output_grad,
                                                      int32_t ignore_index,
                                                      tensor_view_5d_t input_grad_tv,
                                                      tensor_view_4d_t target_tv,
                                                      tensor_view_1d_t weight_tv,
                                                      tensor_view_4d_t output_grad_tv)
{
    nlllossUnreducedBackward5d<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                                        target,
                                                        weight,
                                                        output_grad,
                                                        ignore_index,
                                                        input_grad_tv,
                                                        target_tv,
                                                        weight_tv,
                                                        output_grad_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedBackward4d(TO* __restrict__ input_grad,
                                           const int32_t* __restrict__ target,
                                           const TI* weight,
                                           TI* __restrict__ output_grad,
                                           int32_t ignore_index,
                                           tensor_view_4d_t input_grad_tv,
                                           tensor_view_3d_t target_tv,
                                           tensor_view_1d_t weight_tv,
                                           tensor_view_3d_t output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[3];
    GET_NCD(n[0], n[1], n[2], gid, output_grad_tv);

    if(n[0] >= output_grad_tv.size[0])
        return;

    size_t Tidx = TV3D_IDX(target_tv, n[0], n[1], n[2]);
    size_t Oidx = TV3D_IDX(output_grad_tv, n[0], n[1], n[2]);

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[Tidx];
    size_t Iidx = TV4D_IDX(input_grad_tv, n[0], t, n[1], n[2]);
    size_t Widx = TV1D_IDX(weight_tv, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        input_grad[Iidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[Widx]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[Oidx]);
    FLOAT_ACCUM input_grad_value = CVT_FP32_2ACCUM(-1.0f) * w * grad_val;

    input_grad[Iidx] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossUnreducedBackward4d(INPUT_TYPE* __restrict__ input_grad,
                                                      const int32_t* __restrict__ target,
                                                      const INPUT_TYPE* weight,
                                                      OUTPUT_TYPE* __restrict__ output_grad,
                                                      int32_t ignore_index,
                                                      tensor_view_4d_t input_grad_tv,
                                                      tensor_view_3d_t target_tv,
                                                      tensor_view_1d_t weight_tv,
                                                      tensor_view_3d_t output_grad_tv)
{
    nlllossUnreducedBackward4d<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                                        target,
                                                        weight,
                                                        output_grad,
                                                        ignore_index,
                                                        input_grad_tv,
                                                        target_tv,
                                                        weight_tv,
                                                        output_grad_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedBackward4dContiguous(TO* __restrict__ input_grad,
                                                     const int32_t* __restrict__ target,
                                                     const TI* weight,
                                                     TI* __restrict__ output_grad,
                                                     int32_t ignore_index,
                                                     tensor_view_4d_t input_grad_tv,
                                                     tensor_view_3d_t target_tv,
                                                     tensor_view_1d_t weight_tv,
                                                     tensor_view_3d_t output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    size_t n[3];
    GET_NCD(n[0], n[1], n[2], gid, output_grad_tv);

    if(n[0] >= output_grad_tv.size[0])
        return;

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[gid];
    size_t Iidx = TV4D_IDX(input_grad_tv, n[0], t, n[1], n[2]);

    if(t < 0 || t == ignore_index || t >= C)
    {
        input_grad[Iidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w        = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val = CVT_FLOAT2ACCUM(output_grad[gid]);
    FLOAT_ACCUM input_grad_value = CVT_FP32_2ACCUM(-1.0f) * w * grad_val;

    input_grad[Iidx] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void
NLLLossUnreducedBackward4dContiguous(INPUT_TYPE* __restrict__ input_grad,
                                     const int32_t* __restrict__ target,
                                     const INPUT_TYPE* weight,
                                     OUTPUT_TYPE* __restrict__ output_grad,
                                     int32_t ignore_index,
                                     tensor_view_4d_t input_grad_tv,
                                     tensor_view_3d_t target_tv,
                                     tensor_view_1d_t weight_tv,
                                     tensor_view_3d_t output_grad_tv)
{
    nlllossUnreducedBackward4dContiguous<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                                                  target,
                                                                  weight,
                                                                  output_grad,
                                                                  ignore_index,
                                                                  input_grad_tv,
                                                                  target_tv,
                                                                  weight_tv,
                                                                  output_grad_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedBackward2d(TO* __restrict__ input_grad,
                                           const int32_t* __restrict__ target,
                                           const TI* weight,
                                           TI* __restrict__ output_grad,
                                           int32_t ignore_index,
                                           tensor_view_2d_t input_grad_tv,
                                           tensor_view_1d_t target_tv,
                                           tensor_view_1d_t weight_tv,
                                           tensor_view_1d_t output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= target_tv.size[0])
        return;

    size_t Tidx = TV1D_IDX(target_tv, gid);
    size_t Oidx = TV1D_IDX(output_grad_tv, gid);

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[Tidx];
    size_t Iidx = TV2D_IDX(input_grad_tv, gid, t);
    size_t Widx = TV1D_IDX(weight_tv, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        input_grad[Iidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[Widx]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[Oidx]);
    FLOAT_ACCUM input_grad_value = CVT_FP32_2ACCUM(-1.0f) * w * grad_val;

    input_grad[Iidx] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossUnreducedBackward2d(INPUT_TYPE* __restrict__ input_grad,
                                                      const int32_t* __restrict__ target,
                                                      const INPUT_TYPE* weight,
                                                      OUTPUT_TYPE* __restrict__ output_grad,
                                                      int32_t ignore_index,
                                                      tensor_view_2d_t input_grad_tv,
                                                      tensor_view_1d_t target_tv,
                                                      tensor_view_1d_t weight_tv,
                                                      tensor_view_1d_t output_grad_tv)
{
    nlllossUnreducedBackward2d<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                                        target,
                                                        weight,
                                                        output_grad,
                                                        ignore_index,
                                                        input_grad_tv,
                                                        target_tv,
                                                        weight_tv,
                                                        output_grad_tv);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedBackward2dContiguous(TO* __restrict__ input_grad,
                                                     const int32_t* __restrict__ target,
                                                     const TI* weight,
                                                     TI* __restrict__ output_grad,
                                                     int32_t ignore_index,
                                                     tensor_view_2d_t input_grad_tv,
                                                     tensor_view_1d_t target_tv,
                                                     tensor_view_1d_t weight_tv,
                                                     tensor_view_1d_t output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= target_tv.size[0])
        return;

    int32_t C   = weight_tv.size[0];
    int32_t t   = target[gid];
    size_t Iidx = TV2D_IDX(input_grad_tv, gid, t);

    if(t < 0 || t == ignore_index || t >= C)
    {
        input_grad[Iidx] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w        = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val = CVT_FLOAT2ACCUM(output_grad[gid]);
    FLOAT_ACCUM input_grad_value = CVT_FP32_2ACCUM(-1.0f) * w * grad_val;

    input_grad[Iidx] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void
NLLLossUnreducedBackward2dContiguous(INPUT_TYPE* __restrict__ input_grad,
                                     const int32_t* __restrict__ target,
                                     const INPUT_TYPE* weight,
                                     OUTPUT_TYPE* __restrict__ output_grad,
                                     int32_t ignore_index,
                                     tensor_view_2d_t input_grad_tv,
                                     tensor_view_1d_t target_tv,
                                     tensor_view_1d_t weight_tv,
                                     tensor_view_1d_t output_grad_tv)
{
    nlllossUnreducedBackward2dContiguous<INPUT_TYPE, OUTPUT_TYPE>(input_grad,
                                                                  target,
                                                                  weight,
                                                                  output_grad,
                                                                  ignore_index,
                                                                  input_grad_tv,
                                                                  target_tv,
                                                                  weight_tv,
                                                                  output_grad_tv);
}

__device__ FLOAT_ACCUM warpReduceSum(FLOAT_ACCUM val)
{
    if(warpSize >= 64)
        val += __shfl_down(val, 32);
    if(warpSize >= 32)
        val += __shfl_down(val, 16);
    if(warpSize >= 16)
        val += __shfl_down(val, 8);
    if(warpSize >= 8)
        val += __shfl_down(val, 4);
    if(warpSize >= 4)
        val += __shfl_down(val, 2);
    if(warpSize >= 2)
        val += __shfl_down(val, 1);
    return val;
}

__device__ FLOAT_ACCUM blockReduceSum(FLOAT_ACCUM val)
{
    static __shared__ FLOAT_ACCUM shared[REDUCE_SIZE / warpSize];
    auto lane = threadIdx.x % warpSize;
    auto wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if(lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = threadIdx.x < REDUCE_SIZE / warpSize ? shared[lane] : CVT_FP32_2ACCUM(0.0f);
    if(wid == 0)
        val = warpReduceSum(val);

    return val;
}

template <typename DTYPE>
__device__ void lossSum(const DTYPE* input, DTYPE* output, size_t N)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? CVT_FLOAT2ACCUM(input[gid]) : CVT_FP32_2ACCUM(0.0f);
    val             = blockReduceSum(val);

    if(threadIdx.x == 0)
        output[blockIdx.x] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void
LossSum(const D_TYPE* __restrict__ input, D_TYPE* __restrict__ output, size_t N)
{
    lossSum<D_TYPE>(input, output, N);
}
