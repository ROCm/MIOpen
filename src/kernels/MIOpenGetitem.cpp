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

#include "miopen_cstdint.hpp"
#include "float_types.h"

template <typename TI, typename TO>
__device__ void getitembwd(const TI* __restrict__ dy,
                           const TI* __restrict__ x,
                           const TI* __restrict__ rstd,
                           TO* __restrict__ dw,
                           uint64_t outer_size,
                           uint64_t inner_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    FLOAT_ACCUM sum = static_cast<FLOAT_ACCUM>(0);
    for(uint64_t i = 0; i < outer_size; ++i)
    {
        uint64_t input_idx = i * inner_size + gid;

        FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[i]);
        FLOAT_ACCUM pdy   = dy ? CVT_FLOAT2ACCUM(dy[input_idx]) : 0;

        sum += pdy * CVT_FLOAT2ACCUM(x[input_idx]) * prstd;
    }

    if(dw)
    {
        dw[gid] = CVT_ACCUM2FLOAT(sum);
    }
}

extern "C" __global__ void GetitemBwd(const INPUT_TYPE* __restrict__ dy,
                                      const INPUT_TYPE* __restrict__ x,
                                      const INPUT_TYPE* __restrict__ rstd,
                                      OUTPUT_TYPE* __restrict__ dw,
                                      uint64_t outer_size,
                                      uint64_t inner_size)
{
    // instantiate the kernel
    getitembwd<INPUT_TYPE, OUTPUT_TYPE>(dy, x, rstd, dw, outer_size, inner_size);
}

extern "C" __global__ void GetItemBuildIndices(const INDEX_TYPE* __restrict__ index,
                                               INDEX_TYPE* __restrict__ element_index,
                                               INDEX_TYPE* __restrict__ error,
                                               inte32_t index_dim,
                                               inte32_t num_indices,
                                               inte32_t dim_size,
                                               tensor_view_5d_t index_tv,
                                               uint64_t dim_offset,
                                               uint64_t dim_info_offset,
                                               uint64_t error_offset)
{
    // instantiate the kernel
    getitembwd<INPUT_TYPE, OUTPUT_TYPE>(dy, x, rstd, dw, outer_size, inner_size);
}
