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

template <typename TIO>
__device__ void WhereContiguousBackward_Kernel(
    const char* condition, const TIO* output_grad, TIO* input_grad, TIO* other_grad, size_t size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= size)
        return;

    TIO outgrad = output_grad[gid];
    char cond   = condition[gid];

    if(input_grad)
    {
        input_grad[gid] = outgrad * static_cast<TIO>(cond);
    }
    if(other_grad)
    {
        other_grad[gid] = outgrad * static_cast<TIO>(1 - cond);
    }
}

extern "C" __global__ void WhereContiguousBackward(const char* condition,
                                                   const IO_TYPE* output_grad,
                                                   IO_TYPE* input_grad,
                                                   IO_TYPE* other_grad,
                                                   size_t size)
{
    WhereContiguousBackward_Kernel<IO_TYPE>(condition, output_grad, input_grad, other_grad, size);
}
