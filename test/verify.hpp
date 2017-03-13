#ifndef GUARD_VERIFY_HPP
#define GUARD_VERIFY_HPP

#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>
#include <mlopen/returns.hpp>

namespace mlopen {

// Compute the value of a range
template<class R>
using range_value = typename std::decay<decltype(*std::declval<R>().begin())>::type;

template<class... Ts>
using common_type = typename std::common_type<Ts...>::type;

struct sum_fn
{
    template<class T, class U>
    auto operator()(T x, U y) const MLOPEN_RETURNS(x + y);
};
static constexpr sum_fn sum{};

struct max_fn
{
    template<class T, class U>
    auto operator()(T x, U y) const MLOPEN_RETURNS(x > y ? x : y);
};
static constexpr max_fn max{};

struct abs_diff_fn
{
    template<class T, class U>
    auto operator()(T x, U y) const MLOPEN_RETURNS(std::fabs(x - y));
};

static constexpr abs_diff_fn abs_diff{};

template<class T, class U>
T as(T, U x)
{
    return x;
}

struct float_equal_fn
{
    template<class T>
    static bool apply(T x, T y)
    {
        return 
            std::isfinite(x) and 
            std::isfinite(y) and
            std::nextafter(x, std::numeric_limits<T>::lowest()) <= y and 
            std::nextafter(x, std::numeric_limits<T>::max()) >= y;
    }

    template<class T, class U>
    bool operator()(T x, U y) const
    {
        return float_equal_fn::apply<common_type<T, U>>(x, y);
    }

};

static constexpr float_equal_fn float_equal{};

struct compare_mag_fn
{
    template<class T, class U>
    bool operator()(T x, U y) const
    {
        return std::fabs(x) < std::fabs(y);
    }
};
static constexpr compare_mag_fn compare_mag{};

struct square_diff_fn
{
    template<class T, class U>
    double operator()(T x, U y) const { return (x - y)*(x - y); }
};
static constexpr square_diff_fn square_diff{};

template<class R1>
auto range_distance(R1&& r1) MLOPEN_RETURNS
(std::distance(r1.begin(), r1.end()));

template<class R1>
bool range_zero(R1&& r1)
{
    return std::all_of(r1.begin(), r1.end(), [](float x) { return x == 0.0; });
}

template<class R1, class R2, class T, class Reducer, class Product>
T range_product(R1&& r1, R2&& r2, T state, Reducer r, Product p)
{
    return std::inner_product(r1.begin(), r1.end(), r2.begin(), state, r, p);
}

template<class R1, class R2, class Compare>
std::size_t mismatch_idx(R1&& r1, R2&& r2, Compare compare)
{
    auto p = std::mismatch(r1.begin(), r1.end(), r2.begin(), compare);
    return std::distance(r1.begin(), p.first);
}

template<class R1, class R2>
double max_diff(R1&& r1, R2&& r2)
{
    return range_product(r1, r2, 0.0, max, abs_diff);
}

template<class R1, class R2, class T>
std::size_t mismatch_diff(R1&& r1, R2&& r2, T diff)
{
    return mismatch_idx(r1, r2, std::bind(float_equal, diff, std::bind(abs_diff, std::placeholders::_1, std::placeholders::_2)));
}

template<class R1, class R2>
double rms_range(R1&& r1, R2&& r2)
{
    std::size_t n = range_distance(r1);
    if (n == range_distance(r2)) 
    {
        double square_difference = range_product(r1, r2, 0.0, sum_fn{}, square_diff);
        double mag1 = *std::max_element(r1.begin(), r1.end(), compare_mag);
        double mag2 = *std::max_element(r2.begin(), r2.end(), compare_mag);
        double mag = std::max({std::fabs(mag1), std::fabs(mag2), std::numeric_limits<double>::min()});
        return std::sqrt(square_difference) / (std::sqrt(n)*mag);
    }
    else return std::numeric_limits<range_value<R1>>::max();
}

}

#endif
