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
#ifndef GUARD_KERNELS_MIOPENREDUCECALCULATION_HPP
#define GUARD_KERNELS_MIOPENREDUCECALCULATION_HPP

enum class ReduceCalculationOp_t
{
    Prod = 1,
    Sum,
    First_ = Prod,
    Last_  = Sum,
};

#ifndef __HIP_DEVICE_COMPILE__
static_assert(MIOPEN_REDUCE_CALCULATION_PROD == static_cast<int>(ReduceCalculationOp_t::Prod));
static_assert(MIOPEN_REDUCE_CALCULATION_SUM == static_cast<int>(ReduceCalculationOp_t::Sum));
#endif

template <typename T, ReduceCalculationOp_t op>
struct reduce_func
{
    inline constexpr void calculate(T& a, T b) const;
};

template <typename T>
struct reduce_func<T, ReduceCalculationOp_t::Prod>
{
    inline constexpr void calculate(T& a, T b) const { a *= b; }
};

template <typename T>
struct reduce_func<T, ReduceCalculationOp_t::Sum>
{
    inline constexpr void calculate(T& a, T b) const { a += b; }
};

#endif // GUARD_GUARD_KERNELS_MIOPENREDUCEEXTREME_HPP
