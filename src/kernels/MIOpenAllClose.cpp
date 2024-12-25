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
__device__ void allCloseForward(const T* input1,
                                const T* input2,
                                const float atol,
                                const float rtol,
                                const bool equal_nan,
                                bool* workspace,
                                uint64_t numel,
                                tensor_view_t<5> input1_tv,
                                tensor_view_t<5> input2_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= numel)
        return;

    FLOAT_ACCUM input1_fvalue = CVT_FLOAT2ACCUM(input1[input1_tv.get_tensor_view_idx({gid})]);
    FLOAT_ACCUM input2_fvalue = CVT_FLOAT2ACCUM(input2[input2_tv.get_tensor_view_idx({gid})]);

    bool out;
    bool input1_isnan = std::isnan(input1_fvalue);
    bool input2_isnan = std::isnan(input2_fvalue);

    if(input1_isnan || input2_isnan)
    {
        if(equal_nan == 1 && input1_isnan == true && input2_isnan == true)
        {
            out = true;
        }
        else
        {
            out = false;
        }
    }
    else
    {
        if(std::fabs(input1_fvalue - input2_fvalue) <= (atol + rtol * std::fabs(input2_fvalue)))
        {
            out = true;
        }
        else
        {
            out = false;
        }
    }

    workspace[gid] = out;
}

extern "C" __global__ void AllCloseForward(const D_TYPE* input1,
                                           const D_TYPE* input2,
                                           const float atol,
                                           const float rtol,
                                           const bool equal_nan,
                                           bool* workspace,
                                           uint64_t numel,
                                           tensor_view_t<5> input1_tv,
                                           tensor_view_t<5> input2_tv)
{
    allCloseForward<D_TYPE>(
        input1, input2, atol, rtol, equal_nan, workspace, numel, input1_tv, input2_tv);
}

__device__ void allCloseReduce(const bool* workspace, bool* output, uint64_t numel)
{
    uint64_t gid = blockIdx.x;
    uint64_t lid = threadIdx.x;

    int32_t val = (gid < numel) ? static_cast<int32_t>(workspace[gid]) : 1;

    block_reduce<BinaryOp_t::Prod, LOCAL_SIZE, ReduceThreadDim::X>(val);

    if(lid == 0)
    {
        output[0] = static_cast<bool>(val);
    }
}

extern "C" __global__ void AllCloseReduce(const bool* workspace, bool* output, uint64_t numel)
{
    allCloseReduce(workspace, output, numel);
}
