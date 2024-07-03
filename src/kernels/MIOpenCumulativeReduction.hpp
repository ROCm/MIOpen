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
#ifndef GUARD_KERNELS_MIOPEN_CUMULATIVE_REDUCTIONS_HPP
#define GUARD_KERNELS_MIOPEN_CUMULATIVE_REDUCTIONS_HPP

#include "float_types.h"

enum class CumulativeReductionOp_t
{
    Max  = 1,
    Min  = 2,
    Sum  = 3,
    Prod = 4,
};

#ifndef __HIP_DEVICE_COMPILE__
static_assert(MIOPEN_CUM_MAX == static_cast<int>(CumulativeReductionOp_t::Max));
static_assert(MIOPEN_CUM_MIN == static_cast<int>(CumulativeReductionOp_t::Min));
static_assert(MIOPEN_CUM_SUM == static_cast<int>(CumulativeReductionOp_t::Sum));
static_assert(MIOPEN_CUM_PROD == static_cast<int>(CumulativeReductionOp_t::Prod));
#endif

__device__ inline constexpr void update() {}

template <typename T, typename... Ts>
__device__ inline constexpr void update(T& a, T b, Ts&... c, Ts... d)
{
    a = b;
    update(c..., d...);
}

__device__ inline constexpr bool isgreater() { return false; }

template <typename T, typename... Ts>
__device__ inline constexpr bool isgreater(T& a, T b, Ts&... c, Ts... d)
{
    if(a != b)
        return a > b;
    return isgreater(c..., d...);
}

template <typename T, typename... Ts>
struct reduce_func_base
{
    __device__ virtual inline bool isbetter(const T& /*a*/, const T& /*b*/) const { return false; }
    __device__ virtual inline void combine(T& a, T b) const { a = b; }
    __device__ inline constexpr void
    calculate(const bool keep_greater, T& a, T b, Ts&... c, Ts... d) const
    {
        if(!isbetter(a, b))
        {
            if((isbetter(a, b) != isbetter(b, a)) ||
               (isbetter(a, b) == isbetter(b, a) && isgreater(c..., d...) != keep_greater))
                update(c..., d...);
            combine(a, b);
        }
    }
};

template <CumulativeReductionOp_t OP, typename T, typename... Ts>
struct reduce_func;

template <typename T, typename... Ts>
struct reduce_func<CumulativeReductionOp_t::Max, T, Ts...> : reduce_func_base<T, Ts...>
{
    const FLOAT_ACCUM START_VAL = -MAX_VAL_ACCUM;
    __device__ inline bool isbetter(const T& a, const T& b) const { return a > b; }
};

template <typename T, typename... Ts>
struct reduce_func<CumulativeReductionOp_t::Min, T, Ts...> : reduce_func_base<T, Ts...>
{
    const FLOAT_ACCUM START_VAL = MAX_VAL_ACCUM;
    __device__ inline bool isbetter(const T& a, const T& b) const { return a < b; }
};

template <typename T, typename... Ts>
struct reduce_func<CumulativeReductionOp_t::Sum, T, Ts...> : reduce_func_base<T, Ts...>
{
    const FLOAT_ACCUM START_VAL = CVT_FP32_2ACCUM(0.0f);
    __device__ inline void combine(T& a, T b) const { a += b; }
};

template <typename T, typename... Ts>
struct reduce_func<CumulativeReductionOp_t::Prod, T, Ts...> : reduce_func_base<T, Ts...>
{
    const FLOAT_ACCUM START_VAL = CVT_FP32_2ACCUM(1.0f);
    __device__ inline void combine(T& a, T b) const { a *= b; }
};

#endif // GUARD_GUARD_KERNELS_MIOPEN_CUMULATIVE_REDUCTIONS_HPP
