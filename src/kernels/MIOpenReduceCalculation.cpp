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
#include "MIOpenReduceCalculation.hpp"

template <typename TI, typename TO, ReduceCalculationOp_t op>
__device__ void calculationparallelfwdcontiguous(const TI* __restrict__ x,
                                                 TO* __restrict__ y,
                                                 uint64_t output_numel,
                                                 uint64_t reduce_size,
                                                 uint64_t parallelism_size,
                                                 uint64_t inner_size,
                                                 bool nanPropagation)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= parallelism_size * output_numel)
        return;

    uint64_t n = inner_size * parallelism_size;

    uint64_t slice_id       = gid / n;
    uint64_t slice_local_id = gid % n;

    uint64_t input_idx = slice_id * inner_size * reduce_size + slice_local_id;

    uint64_t parallel_id = slice_local_id / inner_size;

    FLOAT_ACCUM calculation = static_cast<FLOAT_ACCUM>(0);
    for(uint64_t k = parallel_id; k < reduce_size; k += parallelism_size)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(x[input_idx]);
        if(nanPropagation && isnan(val))
        {
            val = static_cast<FLOAT_ACCUM>(0);
        }
        reduce_func<FLOAT_ACCUM, op>{}.calculate(calculation, val);
        input_idx += inner_size * parallelism_size;
    }

    y[gid] = CVT_ACCUM2FLOAT(calculation);
}

extern "C" __global__ void CalculationParallelFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                                            OUTPUT_TYPE* __restrict__ y,
                                                            uint64_t output_numel,
                                                            uint64_t reduce_size,
                                                            uint64_t parallelism_size,
                                                            uint64_t inner_size,
                                                            bool nanPropagation)
{
    // instantiate the kernel
    calculationparallelfwdcontiguous<INPUT_TYPE, OUTPUT_TYPE, OP_TYPE>(
        x, y, output_numel, reduce_size, parallelism_size, inner_size, nanPropagation);
}

template <typename TI, typename TO, ReduceCalculationOp_t op>
__device__ void calculationfwdcontiguous(const TI* __restrict__ x,
                                         TO* __restrict__ y,
                                         uint64_t output_numel,
                                         uint64_t reduce_size,
                                         uint64_t inner_size,
                                         bool nanPropagation)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= output_numel)
        return;

    uint64_t input_idx = (gid / inner_size) * inner_size * reduce_size + gid % inner_size;

    FLOAT_ACCUM calculation = static_cast<FLOAT_ACCUM>(0);
    for(uint64_t k = 0; k < reduce_size; ++k)
    {
        FLOAT_ACCUM val = CVT_FLOAT2ACCUM(x[input_idx]);
        if(nanPropagation && isnan(val))
        {
            val = static_cast<FLOAT_ACCUM>(0);
        }
        reduce_func<FLOAT_ACCUM, op>{}.calculate(calculation, val);
        input_idx += inner_size;
    }

    y[gid] = CVT_ACCUM2FLOAT(calculation);
}

extern "C" __global__ void CalculationFwdContiguous(const INPUT_TYPE* __restrict__ x,
                                                    OUTPUT_TYPE* __restrict__ y,
                                                    uint64_t output_numel,
                                                    uint64_t reduce_size,
                                                    uint64_t inner_size,
                                                    bool nanPropagation)
{
    // instantiate the kernel
    calculationfwdcontiguous<INPUT_TYPE, OUTPUT_TYPE, OP_TYPE>(
        x, y, output_numel, reduce_size, inner_size, nanPropagation);
}