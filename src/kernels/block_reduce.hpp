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
#ifndef GUARD_BLOCK_REDUCE_HPP
#define GUARD_BLOCK_REDUCE_HPP

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "warp_reduce.hpp"

enum class ReduceThreadDim : int32_t
{
    X = 1 << 0,
    Y = 1 << 1,
    Z = 1 << 2,
};

template <BinaryOp_t Op, uint64_t reduce_size, ReduceThreadDim thread_dim>
__device__ FLOAT_ACCUM block_reduce(FLOAT_ACCUM val)
{
    if(reduce_size == warpSize)
        return warp_reduce<Op>(val);

    static __shared__ FLOAT_ACCUM shared[reduce_size / warpSize];
    uint64_t tid = 0;
    if(static_cast<int32_t>(thread_dim) & static_cast<int32_t>(ReduceThreadDim::X))
        tid += threadIdx.x;
    if(static_cast<int32_t>(thread_dim) & static_cast<int32_t>(ReduceThreadDim::Y))
        tid = tid * blockDim.y + threadIdx.y;
    if(static_cast<int32_t>(thread_dim) & static_cast<int32_t>(ReduceThreadDim::Z))
        tid = tid * blockDim.z + threadIdx.z;
    const uint64_t lane = tid % warpSize;
    const uint64_t wid  = tid / warpSize;

    val = warp_reduce<Op>(val);
    if(lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = tid < reduce_size / warpSize ? shared[lane] : 0;
    if(wid == 0)
        val = warp_reduce<Op>(val);
    return val;
}

#endif // GUARD_BLOCK_REDUCE_HPP
