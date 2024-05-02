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

__device__ FLOAT_ACCUM sigmoid(FLOAT_ACCUM x) { return 1.0f / (1.0f + exp(-x)); }

template <typename TI, typename TO>
__device__ void
GLUFwdContiguousKernel(const TI* input, TO* output, long N)
{
    const TI* inputFirstHalf  = input;
    const TI* inputSecondHalf = input + N;
    const size_t gid          = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    FLOAT_ACCUM val1 = CVT_FLOAT2ACCUM(inputFirstHalf[gid]);
    FLOAT_ACCUM val2 = sigmoid(CVT_FLOAT2ACCUM(inputSecondHalf[gid]));
    FLOAT_ACCUM val  = val1 * val2;
    output[gid]      = CVT_ACCUM2FLOAT(val);
}

template <typename TI, typename TO>
__device__ void
GLUBwdContiguousKernel(const TI* input, TI* output_grad, const TO* input_grad, long N)
{
    const TI* inputFirstHalf  = input;
    const TI* inputSecondHalf = input + N;
    TO* inputFirstHalf_grad = input_grad;
    TO* inputSecondHalf_grad = input_grad + N;
    const size_t gid          = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    FLOAT_ACCUM inputFirstHalf_v = CVT_FLOAT_2ACCUM(inputFirstHalf[gid]);
    FLOAT_ACCUM sigmoid_v = sigmoid(CVT_FLOAT2ACCUM(inputSecondHalf[gid]));
    FLOAT_ACCUM grad_v = CVT_FLOAT_2ACCUM(output_grad[gid]);

    inputFirstHalf_grad[gid] = CVT_ACCUM2FLOAT(sigmoid_v * grad_v);
    inputSecondHalf_grad[gid] = CVT_ACCUM2FLOAT((1 - sigmoid_v) * sigmoid_v * grad_v * inputFirstHalf_v);
}

extern "C" __global__ void
GLUFwdContiguous(const INPUT_TYPE* input,  OUTPUT_TYPE* output, long N)
{
    GLUFwdContiguousKernel<INPUT_TYPE, OUTPUT_TYPE>(input, output, N);
}

extern "C" __global__ void
GLUBwdContiguous(const INPUT_TYPE* input, INPUT_TYPE* output_grad, OUTPUT_TYPE* input_grad,  long N)
{
    GLUBwdContiguousKernel<INPUT_TYPE, OUTPUT_TYPE>(input, output_grad, input_grad, N);
}
