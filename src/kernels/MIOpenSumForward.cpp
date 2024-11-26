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
// #include "MIOpenReduceCalculation.hpp"

template <typename TI, typename TO, uint32_t NDIMS>
__device__ void sum_1d_forward(const TI* __restrict__ x,
                               TO* __restrict__ y,
                               uint64_t output_numel,
                               uint64_t reduce_size,
                               uint64_t inner_size,
                               uint64_t reduce_dim,
                               bool nanPropagation,
                               tensor_view_t<NDIMS> input_tv,
                               tensor_view_t<NDIMS - 1> output_tv)
{
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= output_numel)
        return;

    uint64_t idx   = (gid / inner_size) * inner_size * reduce_size + gid % inner_size;
    auto i_tl      = tensor_layout_t<NDIMS>(input_tv, idx);
    auto input_idx = input_tv.get_tensor_view_idx(i_tl);

    FLOAT_ACCUM calculation = static_cast<FLOAT_ACCUM>(0);

    for(uint64_t k = 0; k < reduce_size; ++k)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(x[input_idx]);
        if(nanPropagation && isnan(val))
        {
            val = static_cast<FLOAT_ACCUM>(0);
        }
        calculation += val;
        input_idx += input_tv.stride[reduce_dim];
    }

    auto o_tl       = tensor_layout_t<NDIMS - 1>(output_tv, gid);
    auto output_idx = output_tv.get_tensor_view_idx(o_tl);

    // printf("[MIOpenSumForward] gid: %d, calculation: %f\n", gid, calculation);

    y[output_idx]   = CVT_ACCUM2FLOAT(calculation);
}

extern "C" __global__ void Sum1dForward(const INPUT_TYPE* __restrict__ x,
                                        OUTPUT_TYPE* __restrict__ y,
                                        uint64_t output_numel,
                                        uint64_t reduce_size,
                                        uint64_t inner_size,
                                        uint64_t reduce_dim,
                                        bool nanPropagation,
                                        tensor_view_t<VIEW_DIMS> input_tv,
                                        tensor_view_t<VIEW_DIMS - 1> output_tv)
{
    sum_1d_forward<INPUT_TYPE, OUTPUT_TYPE, VIEW_DIMS>(x,
                                                   y,
                                                   output_numel,
                                                   reduce_size,
                                                   inner_size,
                                                   reduce_dim,
                                                   nanPropagation,
                                                   input_tv,
                                                   output_tv);
}
