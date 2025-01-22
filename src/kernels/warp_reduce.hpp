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
#ifndef GUARD_WARP_REDUCE_HPP
#define GUARD_WARP_REDUCE_HPP

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

enum class BinaryOp_t
{
    Add,
};

template <BinaryOp_t Op, typename T>
struct BinaryFunc;

template <typename T>
struct BinaryFunc<BinaryOp_t::Add, T>
{
    constexpr void exec(T& a, const T& b) { a += b; }
};

template <BinaryOp_t Op, uint32_t ws = warpSize>
__device__ FLOAT_ACCUM warp_reduce(FLOAT_ACCUM val)
{
    for(auto d = ws / 2; d >= 1; d >>= 1)
        BinaryFunc<Op, FLOAT_ACCUM>{}.exec(val, __shfl_down(val, d));
    return val;
}

#endif // GUARD_WARP_REDUCE_HPP
