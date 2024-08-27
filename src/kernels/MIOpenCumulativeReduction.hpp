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

inline constexpr void update() {}
template <typename T, typename... Ts>
inline constexpr void update(T& a, T b, Ts&... c, Ts... d)
{
    a = b;
    update(c..., d...);
}

inline constexpr bool isgreater() { return false; }
template <typename T, typename... Ts>
inline constexpr bool isgreater(T& a, T b, Ts&... c, Ts... d)
{
    if(a != b)
        return a > b;
    return isgreater(c..., d...);
}

template <typename reduce_func_derivedT, typename T, typename... Ts>
struct reduce_func_base
{
    inline constexpr bool isbetter(const T& /*a*/, const T& /*b*/) { return false; }
    inline constexpr void combine(T& a, T b) { a = b; }
    inline constexpr void calculate(const bool keep_greater, T& a, T b, Ts&... c, Ts... d)
    {
        auto derived = static_cast<reduce_func_derivedT*>(this);
        if(!derived->isbetter(a, b))
        {
            if(derived->isbetter(a, b) != derived->isbetter(b, a) ||
               isgreater(c..., d...) != keep_greater)
                update(c..., d...);
            derived->combine(a, b);
        }
    }
};

template <CumulativeReductionOp_t OP, typename T, typename... Ts>
struct reduce_func;

template <typename T, typename... Ts>
struct reduce_func<CumulativeReductionOp_t::Max, T, Ts...>
    : reduce_func_base<reduce_func<CumulativeReductionOp_t::Max, T, Ts...>, T, Ts...>
{
    const FLOAT_ACCUM START_VAL = -MAX_VAL_ACCUM;
    inline constexpr bool isbetter(const T& a, const T& b) { return a > b; }
};

template <typename T, typename... Ts>
struct reduce_func<CumulativeReductionOp_t::Min, T, Ts...>
    : reduce_func_base<reduce_func<CumulativeReductionOp_t::Min, T, Ts...>, T, Ts...>
{
    const FLOAT_ACCUM START_VAL = MAX_VAL_ACCUM;
    inline constexpr bool isbetter(const T& a, const T& b) { return a < b; }
};

template <typename T, typename... Ts>
struct reduce_func<CumulativeReductionOp_t::Sum, T, Ts...>
    : reduce_func_base<reduce_func<CumulativeReductionOp_t::Sum, T, Ts...>, T, Ts...>
{
    const FLOAT_ACCUM START_VAL = CVT_FP32_2ACCUM(0.0f);
    inline constexpr void combine(T& a, T b) { a += b; }
};

template <typename T, typename... Ts>
struct reduce_func<CumulativeReductionOp_t::Prod, T, Ts...>
    : reduce_func_base<reduce_func<CumulativeReductionOp_t::Prod, T, Ts...>, T, Ts...>
{
    const FLOAT_ACCUM START_VAL = CVT_FP32_2ACCUM(1.0f);
    inline constexpr void combine(T& a, T b) { a *= b; }
};

#endif // GUARD_GUARD_KERNELS_MIOPEN_CUMULATIVE_REDUCTIONS_HPP
