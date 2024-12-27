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
#include "block_reduce.hpp"

template <typename T>
__device__ void allCloseForward(const T* __restrict__ input1,
                                const T* __restrict__ input2,
                                int32_t* __restrict__ workspace,
                                const float atol,
                                const float rtol,
                                const bool equal_nan,
                                uint64_t numel,
                                tensor_view_t<5> input1_tv,
                                tensor_view_t<5> input2_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= numel)
        return;

    tensor_layout_t<5> layout(input1_tv, gid);

    FLOAT_ACCUM input1_fvalue = CVT_FLOAT2ACCUM(input1[input1_tv.get_tensor_view_idx(layout)]);
    FLOAT_ACCUM input2_fvalue = CVT_FLOAT2ACCUM(input2[input2_tv.get_tensor_view_idx(layout)]);

    int32_t out;
    bool input1_isnan = isnan(input1_fvalue);
    bool input2_isnan = isnan(input2_fvalue);

    if(input1_isnan || input2_isnan)
    {
        if(equal_nan == 1 && input1_isnan == true && input2_isnan == true)
        {
            out = 1;
        }
        else
        {
            out = 0;
        }
    }
    else
    {
        if(fabs(input1_fvalue - input2_fvalue) <= (atol + rtol * fabs(input2_fvalue)))
        {
            out = 1;
        }
        else
        {
            out = 0;
        }
    }

    workspace[gid] = out;
}

extern "C" __global__ void AllCloseForward(const D_TYPE* __restrict__ input1,
                                           const D_TYPE* __restrict__ input2,
                                           int32_t* __restrict__ workspace,
                                           const float atol,
                                           const float rtol,
                                           const bool equal_nan,
                                           uint64_t numel,
                                           tensor_view_t<5> input1_tv,
                                           tensor_view_t<5> input2_tv)
{
    allCloseForward<D_TYPE>(
        input1, input2, workspace, atol, rtol, equal_nan, numel, input1_tv, input2_tv);
}

template <typename T>
__device__ void reduceProd(const T* input, T* output, uint64_t numel)
{
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < numel ? static_cast<FLOAT_ACCUM>(input[gid]) : 1.0f;
    val             = block_reduce<BinaryOp_t::Prod, REDUCE_SIZE, ReduceThreadDim::X>(val);

    if(threadIdx.x == 0)
    {
        output[blockIdx.x] = static_cast<T>(val);
    }
}

extern "C" __global__ void
ReduceProd(const REDUCE_DTYPE* input, REDUCE_DTYPE* output, uint64_t numel)
{
    reduceProd<REDUCE_DTYPE>(input, output, numel);
}
