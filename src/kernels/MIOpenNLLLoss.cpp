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

template <typename TI, typename TO>
__device__ void nlllossUnreducedForward4dContiguous(const TI* __restrict__ input,
                                                    const int32_t* __restrict__ target,
                                                    const TI* weight,
                                                    TO* __restrict__ output,
                                                    int32_t ignore_index,
                                                    size_t N_total,
                                                    size_t C,
                                                    size_t D1,
                                                    size_t D2)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= N_total)
        return;

    size_t NWH[3];
    NWH[2]    = (gid) % D2;
    size_t nc = (gid) / D2;
    NWH[1]    = nc % D1;
    NWH[0]    = nc / D1;

    int32_t t = target[gid];
    if(t < 0 || t == ignore_index || t >= C)
    {
        output[gid] = static_cast<TO>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);

    uint32_t input_offset   = (NWH[0] * C + t) * D1 * D2 + NWH[1] * D2 + NWH[2];
    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[input_offset]);

    FLOAT_ACCUM val = CVT_FP32_2ACCUM(-1.0f) * w * input_value;
    output[gid]     = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward4dContiguous(const INPUT_TYPE* __restrict__ input,
                                                               const int32_t* __restrict__ target,
                                                               const INPUT_TYPE* weight,
                                                               OUTPUT_TYPE* __restrict__ output,
                                                               int32_t ignore_index,
                                                               size_t N_total,
                                                               size_t C,
                                                               size_t D1,
                                                               size_t D2)
{
    nlllossUnreducedForward4dContiguous<INPUT_TYPE, OUTPUT_TYPE>(
        input, target, weight, output, ignore_index, N_total, C, D1, D2);
}

template <typename TI, typename TO>
__device__ void nlllossUnreducedBackward4dContiguous(TO* __restrict__ input_grad,
                                                     const int32_t* __restrict__ target,
                                                     const TI* weight,
                                                     TI* __restrict__ output_grad,
                                                     int32_t ignore_index,
                                                     size_t N_total,
                                                     size_t C,
                                                     size_t D1,
                                                     size_t D2)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= N_total)
        return;

    size_t NWH[3];
    NWH[2]    = (gid) % D2;
    size_t nc = (gid) / D2;
    NWH[1]    = nc % D1;
    NWH[0]    = nc / D1;

    int32_t t             = target[gid];
    uint32_t input_offset = (NWH[0] * C + t) * D1 * D2 + NWH[1] * D2 + NWH[2];
    if(t < 0 || t == ignore_index || t >= C)
    {
        input_grad[input_offset] = static_cast<TO>(0);
        return;
    }

    // TI: ACCUM
    // TO: FLOAT

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM grad_val = CVT_FLOAT2ACCUM(output_grad[gid]);

    FLOAT_ACCUM input_grad_value   = CVT_FP32_2ACCUM(-1.0f) * w * grad_val;
    input_grad[input_offset] = CVT_ACCUM2FLOAT(input_grad_value);
    // if (input_grad_value != 0.0f) 
    // {
    //     printf("grad_val GPU: %f\n", CVT_FLOAT2ACCUM(grad_val));
    //     printf("output GPU: %f\n", CVT_FLOAT2ACCUM(output_grad[gid]));
    //     printf("weight GPU: %f\n", CVT_FLOAT2ACCUM(weight[t]));
    //     printf("input_grad GPU: %f\n", CVT_FLOAT2ACCUM(input_grad[input_offset]));
    // }
}

extern "C" __global__ void
NLLLossUnreducedBackward4dContiguous(INPUT_TYPE* __restrict__ input_grad,
                                     const int32_t* __restrict__ target,
                                     const INPUT_TYPE* weight,
                                     OUTPUT_TYPE* __restrict__ output_grad,
                                     int32_t ignore_index,
                                     size_t N_total,
                                     size_t C,
                                     size_t D1,
                                     size_t D2)
{
    nlllossUnreducedBackward4dContiguous<INPUT_TYPE, OUTPUT_TYPE>(
        input_grad, target, weight, output_grad, ignore_index, N_total, C, D1, D2);
}