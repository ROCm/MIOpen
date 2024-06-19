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
    Max = 1,
    Min,
    Sum,
    Prod,
};

#ifndef __HIP_DEVICE_COMPILE__
static_assert(MIOPEN_CUM_MAX == static_cast<int>(CumulativeReductionOp_t::Max));
static_assert(MIOPEN_CUM_MIN == static_cast<int>(CumulativeReductionOp_t::Min));
static_assert(MIOPEN_CUM_SUM == static_cast<int>(CumulativeReductionOp_t::Sum));
static_assert(MIOPEN_CUM_PROD == static_cast<int>(CumulativeReductionOp_t::Prod));
#endif

template <CumulativeReductionOp_t op, typename T1, typename... T2>
struct reduce_func
{
    const FLOAT_ACCUM START_VAL;

    inline constexpr bool isbetter(const T1& /*a*/, const T1& /*b*/) const { return true; }
    inline constexpr void combine(T1& a, T1 b) const { a = b; }
    inline constexpr void update(T1& a, T1 b, T2&... ext) const
    {
        a = b;
        update(ext...);
    }
    inline constexpr void calculate(T1& a, T1 b, T2&... ext) const
    {
        if(isbetter(b, a))
        {
            combine(a, b);
            update(ext...);
        }
    }
};

template <typename T>
struct reduce_func<CumulativeReductionOp_t::Max, T>
{
    const FLOAT_ACCUM START_VAL = -MAX_VAL_ACCUM;
    inline constexpr bool isbetter(const T& a, const T& b) const { return a > b; }
};

template <typename T>
struct reduce_func<CumulativeReductionOp_t::Min, T>
{
    const FLOAT_ACCUM START_VAL = MAX_VAL_ACCUM;
    inline constexpr bool isbetter(const T& a, const T& b) const { return a < b; }
};

template <typename T>
struct reduce_func<CumulativeReductionOp_t::Sum, T>
{
    const FLOAT_ACCUM START_VAL = CVT_FP32_2ACCUM(0.0f);
    inline constexpr void combine(T& a, T b) const { a += b; }
};

template <typename T>
struct reduce_func<CumulativeReductionOp_t::Prod, T>
{
    const FLOAT_ACCUM START_VAL = CVT_FP32_2ACCUM(1.0f);
    inline constexpr void combine(T& a, T b) const { a *= b; }
};

#endif // GUARD_GUARD_KERNELS_MIOPEN_CUMULATIVE_REDUCTIONS_HPP
