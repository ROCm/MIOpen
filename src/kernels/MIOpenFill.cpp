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

#include "tensor_view.hpp"

template <typename TIO>
__device__ void fill_zero_contiguous(TIO* output, uint64_t size)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size)
        return;

    output[gid] = static_cast<TIO>(0);
}

extern "C" __global__ void FillZeroContiguous(IO_TYPE* output, uint64_t size)
{
    fill_zero_contiguous<IO_TYPE>(output, size);
}

template <typename TIO, uint32_t NDIMS>
__device__ void fill_zero(TIO* output, const uint64_t size, tensor_view_t<NDIMS> output_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid >= size)
        return;

    auto output_layout                                   = tensor_layout_t<NDIMS>(output_tv, gid);
    output[output_tv.get_tensor_view_idx(output_layout)] = static_cast<TIO>(0);
}

extern "C" __global__ void
FillZero(IO_TYPE* output, const uint64_t size, tensor_view_t<VIEW_DIMS> output_tv)
{
    fill_zero<IO_TYPE, VIEW_DIMS>(output, size, output_tv);
}
