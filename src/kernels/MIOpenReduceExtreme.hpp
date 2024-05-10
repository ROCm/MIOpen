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
#ifndef GUARD_KERNELS_MIOPENREDUCEEXTREME_HPP
#define GUARD_KERNELS_MIOPENREDUCEEXTREME_HPP

enum class ReduceExtremeOp_t
{
    Argmin = 1,
    Argmax,
    Min,
    Max,
    First_ = Argmin,
    Last_  = Max,
};

#ifndef __HIP_DEVICE_COMPILE__
static_assert(MIOPEN_REDUCE_EXTREME_ARGMIN == static_cast<int>(ReduceExtremeOp_t::Argmin));
static_assert(MIOPEN_REDUCE_EXTREME_ARGMAX == static_cast<int>(ReduceExtremeOp_t::Argmax));
static_assert(MIOPEN_REDUCE_EXTREME_MIN == static_cast<int>(ReduceExtremeOp_t::Min));
static_assert(MIOPEN_REDUCE_EXTREME_MAX == static_cast<int>(ReduceExtremeOp_t::Max));
#endif

template <typename T1, typename T2, ReduceExtremeOp_t op>
struct reduce_func
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const;
};

template <typename T1, typename T2>
struct reduce_func<T1, T2, ReduceExtremeOp_t::Max>
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const
    {
        if(a < b)
        {
            a = b;
            c = d;
        }
    }
};

template <typename T1, typename T2>
struct reduce_func<T1, T2, ReduceExtremeOp_t::Min>
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const
    {
        if(a > b)
        {
            a = b;
            c = d;
        }
    }
};

template <typename T1, typename T2>
struct reduce_func<T1, T2, ReduceExtremeOp_t::Argmax>
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const
    {
        if(a < b)
        {
            a = b;
            c = d;
        }
    }
};

template <typename T1, typename T2>
struct reduce_func<T1, T2, ReduceExtremeOp_t::Argmin>
{
    inline constexpr void calculate(T1& a, T1 b, T2& c, T2 d) const
    {
        if(a > b)
        {
            a = b;
            c = d;
        }
    }
};
#endif // GUARD_GUARD_KERNELS_MIOPENREDUCEEXTREME_HPP
