#ifndef GUARD_VERIFY_HPP
#define GUARD_VERIFY_HPP

#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mlopen/returns.hpp>


// Compute the value of a range
template<class R>
using range_value = typename std::decay<decltype(*std::declval<R>().begin())>::type;

struct sum_fn
{
    template<class T, class U>
    auto operator()(T x, U y) const MLOPEN_RETURNS(x + y);
};

struct max_mag_fn
{
    template<class T, class U>
    auto operator()(T x, U y) const MLOPEN_RETURNS(std::max(std::fabs(x), std::fabs(y)));
};

struct square_diff_fn
{
    template<class T, class U>
    auto operator()(T x, U y) const MLOPEN_RETURNS((x - y)*(x - y));
};

template<class R1>
auto range_distance(R1&& r1) MLOPEN_RETURNS
(std::distance(r1.begin(), r1.end()));

template<class R1>
bool range_zero(R1&& r1)
{
    return std::all_of(r1.begin(), r1.end(), [](float x) { return x == 0.0; });
}

template<class R1, class R2>
range_value<R1> rms_range(R1&& r1, R2&& r2)
{
    std::size_t n = range_distance(r1);
    if (n == range_distance(r2)) 
    {
        auto square_diff = std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0, sum_fn{}, square_diff_fn{});
        auto mag = std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0, sum_fn{}, max_mag_fn{});
        return std::sqrt(square_diff / (n*mag));
    }
    else return std::numeric_limits<range_value<R1>>::max();
}

template<class V, class... Ts>
auto verify(V&& v, Ts&&... xs) -> decltype(std::make_pair(v.cpu(xs...), v.gpu(xs...)))
{
    const double tolerance = 10e-6;
    auto out_cpu = v.cpu(xs...);
    auto out_gpu = v.gpu(xs...);
    CHECK(range_distance(out_cpu) == range_distance(out_gpu));
    auto error = rms_range(out_cpu, out_gpu);
    if (error > tolerance)
    {
        std::cout << "FAILED: " << error << std::endl;
        v.fail(error, xs...);
        if (range_zero(out_cpu)) std::cout << "Cpu data is all zeros" << std::endl;
        if (range_zero(out_gpu)) std::cout << "Gpu data is all zeros" << std::endl;
    }
    return std::make_pair(std::move(out_cpu), std::move(out_gpu));
}

#endif
