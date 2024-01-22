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
#include <hip/hip_runtime.h>
#endif

__device__ char* cat_copy_buf(const char* __restrict__ input,
                              char* __restrict__ output,
                              const size_t input_dim_size,
                              const size_t stride)
{
    if(!input)
        return output;

    size_t gid0 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gid1 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t gsz0 = gridDim.x * blockDim.x;
    size_t end  = input_dim_size * stride;
    size_t step = gsz0 * sizeof(ushort4);

    input += gid1 * end;

    size_t i = gid0 * sizeof(ushort4);
    for(; (i + sizeof(ushort4)) <= end; i += step)
        *reinterpret_cast<ushort4*>(output + i) = *reinterpret_cast<const ushort4*>(input + i);

    if((i + sizeof(ushort2)) <= end)
    {
        *reinterpret_cast<ushort2*>(output + i) = *reinterpret_cast<const ushort2*>(input + i);
        i += sizeof(ushort2);
    }

    if(i < end)
        *reinterpret_cast<ushort1*>(output + i) = *reinterpret_cast<const ushort1*>(input + i);

    return output + end;
}

extern "C" __global__ void Cat8FwdPacked(const char* __restrict__ input1,
                                         const char* __restrict__ input2,
                                         const char* __restrict__ input3,
                                         const char* __restrict__ input4,
                                         const char* __restrict__ input5,
                                         const char* __restrict__ input6,
                                         const char* __restrict__ input7,
                                         const char* __restrict__ input8,
                                         char* __restrict__ output,
                                         const size_t input1_dim_size,
                                         const size_t input2_dim_size,
                                         const size_t input3_dim_size,
                                         const size_t input4_dim_size,
                                         const size_t input5_dim_size,
                                         const size_t input6_dim_size,
                                         const size_t input7_dim_size,
                                         const size_t input8_dim_size,
                                         const size_t outer_size,
                                         const size_t stride,
                                         const size_t output_dim_size)
{
    size_t gid = blockIdx.y * blockDim.y + threadIdx.y;

    if(gid >= outer_size)
        return;

    output += gid * output_dim_size * stride;

    output = cat_copy_buf(input1, output, input1_dim_size, stride);
    output = cat_copy_buf(input2, output, input2_dim_size, stride);
    output = cat_copy_buf(input3, output, input3_dim_size, stride);
    output = cat_copy_buf(input4, output, input4_dim_size, stride);
    output = cat_copy_buf(input5, output, input5_dim_size, stride);
    output = cat_copy_buf(input6, output, input6_dim_size, stride);
    output = cat_copy_buf(input7, output, input7_dim_size, stride);
    cat_copy_buf(input8, output, input8_dim_size, stride);
}

extern "C" __global__ void Cat4FwdPacked(const char* __restrict__ input1,
                                         const char* __restrict__ input2,
                                         const char* __restrict__ input3,
                                         const char* __restrict__ input4,
                                         char* __restrict__ output,
                                         const size_t input1_dim_size,
                                         const size_t input2_dim_size,
                                         const size_t input3_dim_size,
                                         const size_t input4_dim_size,
                                         const size_t outer_size,
                                         const size_t stride,
                                         const size_t output_dim_size)
{
    size_t gid = blockIdx.y * blockDim.y + threadIdx.y;

    if(gid >= outer_size)
        return;

    output += gid * output_dim_size * stride;

    output = cat_copy_buf(input1, output, input1_dim_size, stride);
    output = cat_copy_buf(input2, output, input2_dim_size, stride);
    output = cat_copy_buf(input3, output, input3_dim_size, stride);
    cat_copy_buf(input4, output, input4_dim_size, stride);
}

extern "C" __global__ void Cat2FwdPacked(const char* __restrict__ input1,
                                         const char* __restrict__ input2,
                                         char* __restrict__ output,
                                         const size_t input1_dim_size,
                                         const size_t input2_dim_size,
                                         const size_t outer_size,
                                         const size_t stride,
                                         const size_t output_dim_size)
{
    size_t gid = blockIdx.y * blockDim.y + threadIdx.y;

    if(gid >= outer_size)
        return;

    output += gid * output_dim_size * stride;

    output = cat_copy_buf(input1, output, input1_dim_size, stride);
    cat_copy_buf(input2, output, input2_dim_size, stride);
}
