/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_VERIFY_HPP
#define GUARD_VERIFY_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <miopen/float_equal.hpp>
#include <miopen/returns.hpp>
#include <numeric>

namespace miopen {

// Compute the value of a range
template <class R>
using range_value = typename std::decay<decltype(*std::declval<R>().begin())>::type;

struct sum_fn
{
    template <class T, class U>
    auto operator()(T x, U y) const MIOPEN_RETURNS(x + y);
};
static constexpr sum_fn sum{};

struct max_fn
{
    template <class T>
    static T id(T x)
    {
        return x;
    }

    template <class T, class U>
    auto operator()(T x, U y) const MIOPEN_RETURNS(max_fn::id(x > y ? x : y));
};
static constexpr max_fn max{};

struct abs_diff_fn
{
    template <class T, class U>
    auto operator()(T x, U y) const MIOPEN_RETURNS(std::fabs(x - y));
};

static constexpr abs_diff_fn abs_diff{};

struct not_finite_fn
{
    template <class T>
    bool operator()(T x) const
    {
        return not std::isfinite(x);
    }
};
static constexpr not_finite_fn not_finite{};

template <class T, class U>
T as(T, U x)
{
    return x;
}

struct compare_mag_fn
{
    template <class T, class U>
    bool operator()(T x, U y) const
    {
        return std::fabs(x) < std::fabs(y);
    }
};
static constexpr compare_mag_fn compare_mag{};

struct square_diff_fn
{
    template <class T, class U>
    double operator()(T x, U y) const
    {
        return (x - y) * (x - y);
    }
};
static constexpr square_diff_fn square_diff{};

template <class R1>
bool range_empty(R1&& r1)
{
    return r1.begin() == r1.end();
}

template <class R1>
auto range_distance(R1&& r1) MIOPEN_RETURNS(std::distance(r1.begin(), r1.end()));

template <class R1>
bool range_zero(R1&& r1)
{
    return std::all_of(r1.begin(), r1.end(), [](float x) { return x == 0.0; });
}

template <class R1, class R2, class T, class Reducer, class Product>
T range_product(R1&& r1, R2&& r2, T state, Reducer r, Product p)
{
    return std::inner_product(r1.begin(), r1.end(), r2.begin(), state, r, p);
}

template <class R1, class R2, class Compare>
std::size_t mismatch_idx(R1&& r1, R2&& r2, Compare compare)
{
    auto p = std::mismatch(r1.begin(), r1.end(), r2.begin(), compare);
    return std::distance(r1.begin(), p.first);
}

template <class R1, class Predicate>
long find_idx(R1&& r1, Predicate p)
{
    auto it = std::find_if(r1.begin(), r1.end(), p);
    if(it == r1.end())
        return -1;
    else
        return std::distance(r1.begin(), it);
}

template <class R1, class R2>
double max_diff(R1&& r1, R2&& r2)
{
    return range_product(r1, r2, 0.0, max, abs_diff);
}

template <class R1, class R2, class T>
std::size_t mismatch_diff(R1&& r1, R2&& r2, T diff)
{
    return mismatch_idx(
        r1,
        r2,
        std::bind(
            float_equal, diff, std::bind(abs_diff, std::placeholders::_1, std::placeholders::_2)));
}

template <class R1, class R2>
double rms_range(R1&& r1, R2&& r2)
{
    std::size_t n = range_distance(r1);
    if(n == range_distance(r2))
    {
        double square_difference = range_product(r1, r2, 0.0, sum_fn{}, square_diff);
        double mag1              = *std::max_element(r1.begin(), r1.end(), compare_mag);
        double mag2              = *std::max_element(r2.begin(), r2.end(), compare_mag);
        double mag =
            std::max({std::fabs(mag1), std::fabs(mag2), std::numeric_limits<double>::min()});
        return std::sqrt(square_difference) / (std::sqrt(n) * mag);
    }
    else
        return std::numeric_limits<range_value<R1>>::max();
}
} // namespace miopen
#endif
