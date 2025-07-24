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
#include "MIOpenReduceExtreme.hpp"

template <typename TI, typename TO, ReduceExtremeOp_t op>
__device__ void extremefwdcontiguous(const TI* __restrict__ x,
                                     TO* __restrict__ y,
                                     int32_t* __restrict__ indice,
                                     uint64_t output_numel,
                                     int32_t reduce_size,
                                     uint64_t inner_size)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_numel)
        return;

    uint64_t input_idx = (gid / inner_size) * inner_size * reduce_size + gid % inner_size;

    int32_t extreme_idx = 0;
    FLOAT_ACCUM extreme = CVT_FLOAT2ACCUM(x[input_idx]);

    for(int32_t k = 1; k < reduce_size; ++k)
    {
        input_idx += inner_size;
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(x[input_idx]);
        reduce_func<FLOAT_ACCUM, int32_t, op>{}.calculate(extreme, val, extreme_idx, k);
    }
    if(y)
        y[gid] = CVT_ACCUM2FLOAT(extreme);
    indice[gid] = extreme_idx;
}

extern "C" __global__ void ExtremeFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                                OUTPUT_TYPE* __restrict__ y,
                                                int32_t* __restrict__ indice,
                                                uint64_t output_numel,
                                                int32_t reduce_size,
                                                uint64_t inner_size)
{
    // instantiate the kernel
    extremefwdcontiguous<INPUT_TYPE, OUTPUT_TYPE, OP_TYPE>(
        x, y, indice, output_numel, reduce_size, inner_size);
}
