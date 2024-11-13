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
#include <miopen/bfloat16.hpp>
using half         = half_float::half;
using hip_bfloat16 = bfloat16;
#include <hip_float8.hpp>
#include "tensor_holder.hpp"

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

namespace abs_diff_detail {
using std::fabs;
struct fn
{
    template <class T, class U>
    auto operator()(T x, U y) const MIOPEN_RETURNS(fabs(x - y));
};

} // namespace abs_diff_detail

static constexpr abs_diff_detail::fn abs_diff{};

struct not_finite_fn
{
    template <class T, typename std::enable_if<(std::is_floating_point_v<T>), bool>::type = false>
    bool operator()(T x) const
    {
        return !std::isfinite(x);
    }

    template <class T,
              typename std::enable_if<
                  (std::is_same_v<typename std::remove_cv<T>::type, half_float::half>),
                  bool>::type = false>
    bool operator()(T x) const
    {
        return !half_float::isfinite(x);
    }

    template <class T,
              typename std::enable_if<(std::is_same_v<typename std::remove_cv<T>::type, bfloat16>),
                                      bool>::type = false>
    bool operator()(T x) const
    {
        return !std::isfinite(x); // bfloat16 has float() conversion operator
    }

    template <class T, typename std::enable_if<(std::is_integral_v<T>), bool>::type = false>
    bool operator()(T x) const
    {
        std::ignore = x;
        return false;
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
        using std::fabs;
        return fabs(x) < fabs(y);
    }
};
static constexpr compare_mag_fn compare_mag{};

struct square_diff_fn
{
    template <class T, class U>
    double operator()(T x, U y) const
    {
        return static_cast<double>((x - y) * (x - y));
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

template <class T>
bool range_zero(const std::vector<T>& r)
{
    return std::all_of(r.begin(), r.end(), [](T x) { return x == T(); });
}

template <class T>
bool range_zero(const tensor<T>& r)
{
    return range_zero(r.data);
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
int64_t find_idx(R1&& r1, Predicate p)
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

template <class R1, class R2>
auto max_diff_v2(R1&& r1, R2&& r2)
{
    using T            = decltype(r1[0] - r2[0]);
    auto abs_diff_func = [](auto x, auto y) { return x > y ? x - y : y - x; };
    // BUG: deduced wrong datatype, half_float bug
    if constexpr(std::is_same_v<T, half_float::detail::expr>)
        return range_product(r1, r2, half_float::half(), max, abs_diff_func);
    else
        return range_product(r1, r2, T(), max, abs_diff_func);
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
        if(n == 0)
            return 0;
        double square_difference = range_product(r1, r2, 0.0, sum_fn{}, square_diff);
        double mag1 = static_cast<double>(*std::max_element(r1.begin(), r1.end(), compare_mag));
        double mag2 = static_cast<double>(*std::max_element(r2.begin(), r2.end(), compare_mag));
        double mag =
            std::max({std::fabs(mag1), std::fabs(mag2), std::numeric_limits<double>::min()});
        return std::sqrt(square_difference) / (std::sqrt(n) * mag);
    }
    else
        return double(std::numeric_limits<range_value<R1>>::max());
}
} // namespace miopen
#endif
