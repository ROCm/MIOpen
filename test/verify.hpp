#ifndef GUARD_VERIFY_HPP
#define GUARD_VERIFY_HPP

#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

struct float_equal_fn
{
    template<class T>
    bool operator()(T x, T y) const
    {
        return std::fabs(x - y) < std::max(std::numeric_limits<T>::epsilon() * std::max(x, y), std::numeric_limits<T>::epsilon());
    }
};

template<class R1, class R2>
bool float_equal_range(R1&& r1, R2&& r2)
{
    return std::distance(r1.begin(), r1.end()) == std::distance(r2.begin(), r2.end()) &&
        std::equal(r1.begin(), r1.end(), r2.begin(), float_equal_fn{});
}

template<class R1, class R2, class Op>
float accumulate_difference(R1&& r1, R2&& r2, Op op)
{
    return std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0, op,
        [](float x, float y) { return std::fabs(x - y); }
    );
}

template<class V, class... Ts>
void verify(V&& v, Ts&&... xs)
{
    auto out_cpu = v.cpu(xs...);
    auto out_gpu = v.gpu(xs...);
    int size = std::distance(out_cpu.begin(), out_cpu.end());
    CHECK(std::distance(out_cpu.begin(), out_cpu.end()) == std::distance(out_gpu.begin(), out_gpu.end()));
    if (!float_equal_range(out_cpu, out_gpu))
    {
        std::cout << "FAILED: " << std::endl;
        v.fail(xs...);

        std::cout 
            << "Average difference: " 
            << (accumulate_difference(out_cpu, out_gpu, std::plus<float>()) / size) 
            << std::endl;
        std::cout 
            << "Max difference: " 
            << (accumulate_difference(out_cpu, out_gpu, [](float x, float y) { return std::max(x, y); })) 
            << std::endl;
    }
}

#endif
