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

//#if MIOPEN_USE_BFP16 == 1
//#undef FLOAT
//#define FLOAT hip_bfloat16
//#endif

template <typename T>
__device__ void cat_copy_buf(const T* __restrict__ input,
                             T* __restrict__ output,
                             const size_t input_dim_size,
                             const int stride,
                             size_t* output_offset)
{
    if(!input)
        return;

    int gid0            = blockIdx.x * blockDim.x + threadIdx.x;
    int gid1            = blockIdx.y * blockDim.y + threadIdx.y;
    int gsz0            = gridDim.x * blockDim.x;
    size_t input_offset = static_cast<size_t>(gid1) * input_dim_size * static_cast<size_t>(stride);

    size_t end = input_dim_size * static_cast<size_t>(stride);
    for(size_t i = static_cast<size_t>(gid0); i < end; i += static_cast<size_t>(gsz0))
    {
        output[*output_offset + i] = input[input_offset + i];
    }
    *output_offset += input_dim_size * stride;
}

extern "C" __global__ void Cat8FwdPacked(const FLOAT* __restrict__ input1,
                                         const FLOAT* __restrict__ input2,
                                         const FLOAT* __restrict__ input3,
                                         const FLOAT* __restrict__ input4,
                                         const FLOAT* __restrict__ input5,
                                         const FLOAT* __restrict__ input6,
                                         const FLOAT* __restrict__ input7,
                                         const FLOAT* __restrict__ input8,
                                         FLOAT* __restrict__ output,
                                         const size_t input1_dim_size,
                                         const size_t input2_dim_size,
                                         const size_t input3_dim_size,
                                         const size_t input4_dim_size,
                                         const size_t input5_dim_size,
                                         const size_t input6_dim_size,
                                         const size_t input7_dim_size,
                                         const size_t input8_dim_size,
                                         const int dim,
                                         const size_t outer_size,
                                         const int stride,
                                         const size_t output_dim_size)
{
    int gid = blockIdx.y * blockDim.y + threadIdx.y;

    if(gid >= outer_size)
        return;

    size_t output_offset = static_cast<size_t>(gid) * output_dim_size * static_cast<size_t>(stride);

    cat_copy_buf(input1, output, input1_dim_size, stride, &output_offset);
    cat_copy_buf(input2, output, input2_dim_size, stride, &output_offset);
    cat_copy_buf(input3, output, input3_dim_size, stride, &output_offset);
    cat_copy_buf(input4, output, input4_dim_size, stride, &output_offset);
    cat_copy_buf(input5, output, input5_dim_size, stride, &output_offset);
    cat_copy_buf(input6, output, input6_dim_size, stride, &output_offset);
    cat_copy_buf(input7, output, input7_dim_size, stride, &output_offset);
    cat_copy_buf(input8, output, input8_dim_size, stride, &output_offset);
}

extern "C" __global__ void Cat4FwdPacked(const FLOAT* __restrict__ input1,
                                         const FLOAT* __restrict__ input2,
                                         const FLOAT* __restrict__ input3,
                                         const FLOAT* __restrict__ input4,
                                         FLOAT* __restrict__ output,
                                         const size_t input1_dim_size,
                                         const size_t input2_dim_size,
                                         const size_t input3_dim_size,
                                         const size_t input4_dim_size,
                                         const int dim,
                                         const size_t outer_size,
                                         const int stride,
                                         const size_t output_dim_size)
{
    int gid = blockIdx.y * blockDim.y + threadIdx.y;

    if(gid >= outer_size)
        return;

    size_t output_offset = static_cast<size_t>(gid) * output_dim_size * static_cast<size_t>(stride);

    cat_copy_buf(input1, output, input1_dim_size, stride, &output_offset);
    cat_copy_buf(input2, output, input2_dim_size, stride, &output_offset);
    cat_copy_buf(input3, output, input3_dim_size, stride, &output_offset);
    cat_copy_buf(input4, output, input4_dim_size, stride, &output_offset);
}

extern "C" __global__ void Cat2FwdPacked(const FLOAT* __restrict__ input1,
                                         const FLOAT* __restrict__ input2,
                                         FLOAT* __restrict__ output,
                                         const size_t input1_dim_size,
                                         const size_t input2_dim_size,
                                         const int dim,
                                         const size_t outer_size,
                                         const int stride,
                                         const size_t output_dim_size)
{
    int gid = blockIdx.y * blockDim.y + threadIdx.y;

    if(gid >= outer_size)
        return;

    size_t output_offset = static_cast<size_t>(gid) * output_dim_size * static_cast<size_t>(stride);

    cat_copy_buf(input1, output, input1_dim_size, stride, &output_offset);
    cat_copy_buf(input2, output, input2_dim_size, stride, &output_offset);
}
