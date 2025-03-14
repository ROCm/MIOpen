/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

template <typename DTYPE, int REDUCTION_T>
__device__ void hingeembeddinglossforward(const DTYPE* __restrict__ input,
                                          const char* __restrict__ target,
                                          void* __restrict__ output,
                                          const size_t num_elem,
                                          const float margin,
                                          tensor_view_t<5> input_tv,
                                          tensor_view_t<5> target_tv,
                                          tensor_view_t<5> output_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    tensor_layout_t<5> idx(input_tv, gid);
    if(idx.layout[0] >= input_tv.size[0])
        return;

    FLOAT_ACCUM loss;
    if(target[target_tv.get_tensor_view_idx(idx)] == 1)
        loss = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(idx)]);
    else
        loss = fmaxf(0.0f, margin - CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(idx)]));

    switch(REDUCTION_T)
    {
    case 0:
        static_cast<DTYPE*>(output)[output_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(loss);
        break;
    case 1: static_cast<FLOAT_ACCUM*>(output)[gid] = loss; break;
    case 2: static_cast<FLOAT_ACCUM*>(output)[gid] = loss / num_elem; break;
    default: break;
    }
}

extern "C" __global__ void HingeEmbeddingLossForward(const FLOAT* __restrict__ input,
                                                     const char* __restrict__ target,
                                                     void* __restrict__ output,
                                                     const size_t num_elem,
                                                     const float margin,
                                                     tensor_view_t<5> input_tv,
                                                     tensor_view_t<5> target_tv,
                                                     tensor_view_t<5> output_tv)
{
    hingeembeddinglossforward<FLOAT, REDUCTION_TYPE>(
        input, target, output, num_elem, margin, input_tv, target_tv, output_tv);
}

template <typename DTYPE, int REDUCTION_T>
__device__ void hingeembeddinglossbackward(const DTYPE* __restrict__ input,
                                           const char* __restrict__ target,
                                           const DTYPE* __restrict__ doutput,
                                           DTYPE* __restrict__ dinput,
                                           const size_t num_elem,
                                           const float margin,
                                           tensor_view_t<5> input_tv,
                                           tensor_view_t<5> target_tv,
                                           tensor_view_t<5> doutput_tv,
                                           tensor_view_t<5> dinput_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    tensor_layout_t<5> idx(input_tv, gid);
    if(idx.layout[0] >= input_tv.size[0])
        return;

    // TODO: Can optimize perf by vectorization and use shared memory to access doutput[0] for case
    // != reduction_none
    if(target[target_tv.get_tensor_view_idx(idx)] == 1)
    {
        switch(REDUCTION_T)
        {
        case 0:
            dinput[dinput_tv.get_tensor_view_idx(idx)] =
                doutput[doutput_tv.get_tensor_view_idx(idx)];
            break;
        case 1: dinput[dinput_tv.get_tensor_view_idx(idx)] = doutput[0]; break;
        case 2:
            dinput[dinput_tv.get_tensor_view_idx(idx)] =
                CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(doutput[0]) / num_elem);
            break;
        default: break;
        }
    }
    else
    {
        if(margin - CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(idx)]) > 0)
        {
            switch(REDUCTION_T)
            {
            case 0:
                dinput[dinput_tv.get_tensor_view_idx(idx)] =
                    CVT_ACCUM2FLOAT(-CVT_FLOAT2ACCUM(doutput[doutput_tv.get_tensor_view_idx(idx)]));
                break;
            case 1:
                dinput[dinput_tv.get_tensor_view_idx(idx)] =
                    CVT_ACCUM2FLOAT(-CVT_FLOAT2ACCUM(doutput[0]));
                break;
            case 2:
                dinput[dinput_tv.get_tensor_view_idx(idx)] =
                    CVT_ACCUM2FLOAT(-CVT_FLOAT2ACCUM(doutput[0]) / num_elem);
                break;
            default: break;
            }
        }
        else
        {
            dinput[dinput_tv.get_tensor_view_idx(idx)] = 0;
        }
    }
}

extern "C" __global__ void HingeEmbeddingLossBackward(const FLOAT* __restrict__ input,
                                                      const char* __restrict__ target,
                                                      const FLOAT* __restrict__ doutput,
                                                      FLOAT* __restrict__ dinput,
                                                      const size_t num_elem,
                                                      const float margin,
                                                      tensor_view_t<5> input_tv,
                                                      tensor_view_t<5> target_tv,
                                                      tensor_view_t<5> doutput_tv,
                                                      tensor_view_t<5> dinput_tv)
{
    hingeembeddinglossbackward<FLOAT, REDUCTION_TYPE>(input,
                                                      target,
                                                      doutput,
                                                      dinput,
                                                      num_elem,
                                                      margin,
                                                      input_tv,
                                                      target_tv,
                                                      doutput_tv,
                                                      dinput_tv);
}
