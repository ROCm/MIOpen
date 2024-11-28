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

template <typename TIO>
__device__ void resourceApplyGradientDescent(const TIO* __restrict__ var_in,
                                             TIO* __restrict__ var_out,
                                             const TIO* __restrict__ alpha_in,
                                             const TIO* __restrict__ delta_in,
                                             const uint64_t input_size,
                                             tensor_view_t<5> var_in_tv,
                                             tensor_view_t<5> var_out_tv,
                                             tensor_view_t<1> alpha_in_tv,
                                             tensor_view_t<5> delta_in_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_size)
        return;
    auto tensor_layout = tensor_layout_t<5>(var_in_tv, gid);

    FLOAT_ACCUM var   = CVT_FLOAT2ACCUM(var_in[var_in_tv.get_tensor_view_idx(tensor_layout)]);
    FLOAT_ACCUM alpha = CVT_FLOAT2ACCUM(alpha_in[alpha_in_tv.get_tensor_view_idx({0})]);
    FLOAT_ACCUM delta = CVT_FLOAT2ACCUM(delta_in[delta_in_tv.get_tensor_view_idx(tensor_layout)]);

    var -= alpha * delta;

    var_out[var_out_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(var);
}

extern "C" __global__ void ResourceApplyGradientDescent(const D_TYPE* __restrict__ var_in,
                                                        D_TYPE* __restrict__ var_out,
                                                        const D_TYPE* __restrict__ alpha_in,
                                                        const D_TYPE* __restrict__ delta_in,
                                                        const uint64_t input_size,
                                                        tensor_view_t<5> var_in_tv,
                                                        tensor_view_t<5> var_out_tv,
                                                        tensor_view_t<1> alpha_in_tv,
                                                        tensor_view_t<5> delta_in_tv)
{
    resourceApplyGradientDescent<D_TYPE>(var_in,
                                         var_out,
                                         alpha_in,
                                         delta_in,
                                         input_size,
                                         var_in_tv,
                                         var_out_tv,
                                         alpha_in_tv,
                                         delta_in_tv);
}

template <typename TIO>
__device__ void resourceApplyGradientDescentContiguous(const TIO* __restrict__ var_in,
                                                       TIO* __restrict__ var_out,
                                                       const TIO* __restrict__ alpha_in,
                                                       const TIO* __restrict__ delta_in,
                                                       const uint64_t input_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_size)
        return;

    FLOAT_ACCUM var   = CVT_FLOAT2ACCUM(var_in[gid]);
    FLOAT_ACCUM alpha = CVT_FLOAT2ACCUM(alpha_in[0]);
    FLOAT_ACCUM delta = CVT_FLOAT2ACCUM(delta_in[gid]);

    var -= alpha * delta;

    var_out[gid] = CVT_ACCUM2FLOAT(var);
}

extern "C" __global__ void
ResourceApplyGradientDescentContiguous(const D_TYPE* __restrict__ var_in,
                                       D_TYPE* __restrict__ var_out,
                                       const D_TYPE* __restrict__ alpha_in,
                                       const D_TYPE* __restrict__ delta_in,
                                       const uint64_t input_size)
{
    resourceApplyGradientDescentContiguous<D_TYPE>(var_in, var_out, alpha_in, delta_in, input_size);
}
