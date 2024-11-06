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

template <typename T>
__device__ void cartesianProdForward(const T* input,
                                     T* output_ws,
                                     tensor_view_t<1> input_tv,
                                     tensor_view_t<2> output_tv,
                                     uint64_t stride,
                                     uint64_t dim1_idx)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_tv.size[0])
        return;
    output_ws[output_tv.size[0] * dim1_idx + gid] =
        input[input_tv.get_tensor_view_idx({(gid / stride) % input_tv.size[0]})];
}

extern "C" __global__ void CartesianProdForward(const D_TYPE* __restrict__ input,
                                                D_TYPE* __restrict__ output_ws,
                                                tensor_view_t<1> input_tv,
                                                tensor_view_t<2> output_tv,
                                                uint64_t stride,
                                                uint64_t dim1_idx)
{
    cartesianProdForward<D_TYPE>(input, output_ws, input_tv, output_tv, stride, dim1_idx);
}

template <typename T>
__device__ void cartesianProdTranspose(const T* output_ws, T* output, tensor_view_t<2> output_tv)
{
    __shared__ T block[TILE_SIZE][TILE_SIZE + 1];

    uint64_t lid[2];
    lid[0] = threadIdx.x;
    lid[1] = threadIdx.y;

    uint64_t transposed_n[2];
    transposed_n[1] = blockIdx.x * TILE_SIZE + lid[0];
    transposed_n[0] = blockIdx.y * TILE_SIZE + lid[1];

    if((transposed_n[1] < output_tv.size[0]) && (transposed_n[0] < output_tv.size[1]))
    {
        uint64_t workspace_idx = transposed_n[0] * output_tv.size[0] + transposed_n[1];
        block[lid[1]][lid[0]]  = output_ws[workspace_idx];
    }

    __syncthreads();

    uint64_t output_n[2];
    output_n[1] = blockIdx.y * TILE_SIZE + lid[0];
    output_n[0] = blockIdx.x * TILE_SIZE + lid[1];

    if((output_n[1] < output_tv.size[1]) && (output_n[0] < output_tv.size[0]))
    {
        output[output_tv.get_tensor_view_idx({output_n[0], output_n[1]})] = block[lid[0]][lid[1]];
    }
}

extern "C" __global__ void CartesianProdTranspose(const D_TYPE* __restrict__ output_ws,
                                                  D_TYPE* __restrict__ output,
                                                  tensor_view_t<2> output_tv)
{
    cartesianProdTranspose<D_TYPE>(output_ws, output, output_tv);
}

template <typename T>
__device__ void cartesianProdBackward(const T* __restrict__ output_grad,
                                      T* __restrict__ input_grad,
                                      tensor_view_t<2> output_grad_tv,
                                      tensor_view_t<1> input_grad_tv,
                                      uint64_t stride,
                                      uint64_t dim1_idx)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_grad_tv.size[0])
        return;

    FLOAT_ACCUM sum = 0;
    for(uint64_t offset = 0; offset < output_grad_tv.size[0];
        offset += (stride * input_grad_tv.size[0]))
    {
        for(uint64_t i = 0; i < stride; ++i)
        {
            uint64_t dim0_idx = offset + gid * stride + i;
            sum += CVT_FLOAT2ACCUM(
                output_grad[output_grad_tv.get_tensor_view_idx({dim0_idx, dim1_idx})]);
        }
    }

    input_grad[input_grad_tv.get_tensor_view_idx({gid})] = CVT_ACCUM2FLOAT(sum);
}

extern "C" __global__ void CartesianProdBackward(const D_TYPE* __restrict__ output_grad,
                                                 D_TYPE* __restrict__ input_grad,
                                                 tensor_view_t<2> output_grad_tv,
                                                 tensor_view_t<1> input_grad_tv,
                                                 uint64_t stride,
                                                 uint64_t dim1_idx)
{
    cartesianProdBackward<D_TYPE>(
        output_grad, input_grad, output_grad_tv, input_grad_tv, stride, dim1_idx);
}
