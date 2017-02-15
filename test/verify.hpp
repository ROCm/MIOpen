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

template<class T, class U>
T as(T, U x)
{
    return x;
}

struct float_equal_fn
{
    template<class T>
    bool operator()(T x, T y) const
    {
        return std::nextafter(x, std::numeric_limits<T>::lowest()) <= y && std::nextafter(x, std::numeric_limits<T>::max()) >= y;
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

struct square_diff_fn
{
    template<class T, class U>
    double operator()(T x, U y) const { return (x - y)*(x - y); }
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
double rms_range(R1&& r1, R2&& r2)
{
    std::size_t n = range_distance(r1);
    if (n == range_distance(r2)) 
    {
        double square_diff = std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0, sum_fn{}, square_diff_fn{});
        double mag1 = *std::max_element(r1.begin(), r1.end(), compare_mag_fn{});
        double mag2 = *std::max_element(r2.begin(), r2.end(), compare_mag_fn{});
        double mag = std::max({std::fabs(mag1), std::fabs(mag2)});
        return std::sqrt(square_diff) / (std::sqrt(n)*mag);
    }
    else return std::numeric_limits<range_value<R1>>::max();
}

template<class V, class... Ts>
auto verify(V&& v, Ts&&... xs) -> decltype(std::make_pair(v.cpu(xs...), v.gpu(xs...)))
{
    // v.fail(0.0, xs...);
    try {
        auto out_cpu = v.cpu(xs...);
        auto out_gpu = v.gpu(xs...);
        CHECK(range_distance(out_cpu) == range_distance(out_gpu));
        
        // using value_type = range_value<decltype(out_gpu)>;
        // const double tolerance = std::numeric_limits<value_type>::epsilon() * 4;
        const double tolerance = 10e-6;
        auto error = rms_range(out_cpu, out_gpu);
        if (error <= tolerance)
        {
            std::cout << "FAILED: " << error << std::endl;
            v.fail(error, xs...);
            if (range_zero(out_cpu)) std::cout << "Cpu data is all zeros" << std::endl;
            if (range_zero(out_gpu)) std::cout << "Gpu data is all zeros" << std::endl;
            auto p = std::mismatch(out_cpu.begin(), out_cpu.end(), out_gpu.begin(), float_equal);
            auto idx = std::distance(out_cpu.begin(), p.first);
            std::cout << "Mismatch at " << idx << ": " << out_cpu[idx] << " != " << out_gpu[idx] << std::endl;
        }
        return std::make_pair(std::move(out_cpu), std::move(out_gpu));
    } catch(const std::exception& ex) {
        std::cout << "FAILED: " << ex.what() << std::endl;
        v.fail(0.0, xs...);
        throw;
    } catch(...) {
        std::cout << "FAILED with unknown exception" << std::endl;
        v.fail(0.0, xs...);
        throw;
    }
}

#endif
