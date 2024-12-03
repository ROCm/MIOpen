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
#pragma once

enum class ReduceCalculationOp_t
{
    First_ = 1,
    Prod   = First_,
    Sum,
    lOR,  // Logical OR, to distinguish from bitwise OR
    lAND, // Logical AND, to distinguish from bitwise AND
    Last_ = lAND,
};

#ifndef __HIP_DEVICE_COMPILE__
static_assert(MIOPEN_REDUCE_CALCULATION_PROD == static_cast<int>(ReduceCalculationOp_t::Prod));
static_assert(MIOPEN_REDUCE_CALCULATION_SUM == static_cast<int>(ReduceCalculationOp_t::Sum));
static_assert(MIOPEN_REDUCE_CALCULATION_ANY == static_cast<int>(ReduceCalculationOp_t::lOR));
static_assert(MIOPEN_REDUCE_CALCULATION_ALL == static_cast<int>(ReduceCalculationOp_t::lAND));
#endif

template <typename T, ReduceCalculationOp_t op>
struct reduce_func
{
    inline constexpr void calculate(T& a, T b) const;
    inline constexpr T get_initial_value() const;
};

template <typename T>
struct reduce_func<T, ReduceCalculationOp_t::Prod>
{
    inline constexpr void calculate(T& a, T b) const { a *= b; }
    inline constexpr T get_initial_value() const { return static_cast<T>(1); }
};

template <typename T>
struct reduce_func<T, ReduceCalculationOp_t::Sum>
{
    inline constexpr void calculate(T& a, T b) const { a += b; }
    inline constexpr T get_initial_value() const { return static_cast<T>(0); }
};

template <typename T>
struct reduce_func<T, ReduceCalculationOp_t::lOR>
{
    inline constexpr void calculate(T& a, T b) const { a = a || b; }
    inline constexpr T get_initial_value() const { return static_cast<T>(0); }
};

template <typename T>
struct reduce_func<T, ReduceCalculationOp_t::lAND>
{
    inline constexpr void calculate(T& a, T b) const { a = a && b; }
    inline constexpr T get_initial_value() const { return static_cast<T>(1); }
};
