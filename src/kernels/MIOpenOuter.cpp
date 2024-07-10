/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <stdio.h>

extern "C" __global__ void OuterForward(const FLOAT* input1,
                                        const FLOAT* input2,
                                        FLOAT* output,
                                        const size_t n,
                                        const size_t m,
                                        const size_t nm)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= nm)
        return;

    size_t ix[2];

    ix[0] = gid / m;
    ix[1] = gid % m;

    output[gid] = CVT_ACCUM2FLOAT(CVT_FLOAT2ACCUM(input1[ix[0]]) * CVT_FLOAT2ACCUM(input2[ix[1]]));
}

extern "C" __global__ void OuterBackwardGrad1(const FLOAT* input2,
                                              FLOAT* input1_grad,
                                              const FLOAT* output_grad,
                                              const size_t n,
                                              const size_t m)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= n)
        return;

    FLOAT_ACCUM sum = 0;
    for(unsigned int j = 0; j < m; ++j)
    {
        sum += CVT_FLOAT2ACCUM(input2[j]) * CVT_FLOAT2ACCUM(output_grad[gid * m + j]);
    }

    input1_grad[gid] = CVT_ACCUM2FLOAT(sum);
}

extern "C" __global__ void OuterBackwardGrad2(const FLOAT* input1,
                                              FLOAT* input2_grad,
                                              const FLOAT* output_grad,
                                              const size_t n,
                                              const size_t m)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= m)
        return;

    FLOAT_ACCUM sum = 0;
    for(unsigned int i = 0; i < n; ++i)
    {
        sum += CVT_FLOAT2ACCUM(input1[i]) * CVT_FLOAT2ACCUM(output_grad[i * m + gid]);
    }

    input2_grad[gid] = CVT_ACCUM2FLOAT(sum);
}
