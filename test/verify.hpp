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
    auto operator()(T x, U y) const MLOPEN_RETURNS
    (
        std::max(std::max(std::fabs(x), std::fabs(y)), static_cast<decltype(std::max(std::fabs(x), std::fabs(y)))>(1))
    );
};

struct square_diff_fn
{
    template<class T, class U>
    auto operator()(T x, U y) const MLOPEN_RETURNS((x - y)*(x - y));
};

template<class R1, class R2>
range_value<R1> rms_range(R1&& r1, R2&& r2)
{
    std::size_t n = std::distance(r1.begin(), r1.end());
    if (n == std::distance(r2.begin(), r2.end())) 
    {
        auto square_diff = std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0, sum_fn{}, square_diff_fn{});
        auto mag = std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0, sum_fn{}, max_mag_fn{});
        return std::sqrt(square_diff / (n*mag));
    }
    else return std::numeric_limits<range_value<R1>>::max();
}

template<class V, class... Ts>
void verify(V&& v, Ts&&... xs)
{
    const double tolerance = 10e-6;
    auto out_cpu = v.cpu(xs...);
    auto out_gpu = v.gpu(xs...);
    CHECK(std::distance(out_cpu.begin(), out_cpu.end()) == std::distance(out_gpu.begin(), out_gpu.end()));
    auto error = rms_range(out_cpu, out_gpu);
    if (error > tolerance)
    {
        std::cout << "FAILED: " << error << std::endl;
        v.fail(error, xs...);
        // TODO: Check if gpu data is all zeros
    }
}

#endif
