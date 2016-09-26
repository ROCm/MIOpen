#ifndef GUARD_FORD_HPP
#define GUARD_FORD_HPP

#include <mlopen/returns.hpp>
#include <mlopen/each_args.hpp>
#include <functional>
#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>
#include <cmath>
#include <cassert>

template<class F>
struct protect_fn
{
    F f;
    protect_fn(F x) : f(std::move(x)) 
    {}

    template<class... Ts>
    auto operator()(Ts&&... xs) const MLOPEN_RETURNS
    (f(std::forward<Ts>(xs)...));
};

template<class F>
protect_fn<F> protect(F f)
{
    return {std::move(f)};
}

template<class F>
void par_for(std::size_t n, F f)
{
    const auto threadsize = std::thread::hardware_concurrency();
    if (n < threadsize)
    {
        for(std::size_t i=0;i<n;i++) f(i);
    }
    else
    {
        std::vector<std::thread> threads(threadsize);
        const std::size_t grainsize = std::ceil(static_cast<double>(n) / threads.size());

        std::size_t work = 0;
        std::generate(threads.begin(), threads.end(), [&]
        {
            auto result = std::thread([&, work]
            {
                std::size_t start = work;
                std::size_t last = std::min(n, work+grainsize);
                for(std::size_t i=start;i<last;i++) 
                {
                    f(i);
                }
            });
            work += grainsize;
            return result;
        });
        assert(work >= n);
        // TODO: Should be in destructor
        for(auto&& t:threads)
        {
            if (t.joinable()) t.join();
        }
    }
}

// Multidimensional for loop
struct ford_impl
{
    template<class F>
    void operator()(F f) const
    {
        f();
    }

    template<class F, class T, class... Ts>
    void operator()(F f, T x, Ts... xs) const
    {
        // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55914
        for(T i=0;i<x;i++)
        {
            (*this)([&](Ts... is)
            {
                f(i, is...);
            }, xs...);
        }
    }
};

template<class... Ts>
auto ford(Ts... xs) MLOPEN_RETURNS
(
    std::bind(ford_impl{}, std::placeholders::_1, xs...)
);

struct par_ford_impl
{
    template<class F, class... Ts>
    void operator()(F f, Ts... xs) const
    {
        using array_type = std::array<std::size_t, sizeof...(Ts)>;
        array_type lens = {{static_cast<std::size_t>(xs)...}};
        array_type strides;
        strides.fill(1);
        std::partial_sum(lens.rbegin(), lens.rend()-1, strides.rbegin()+1, std::multiplies<std::size_t>());
        auto size = std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<std::size_t>());
        par_for(size, [&](std::size_t i)
        {
            array_type indices;
            std::transform(strides.begin(), strides.end(), lens.begin(), indices.begin(), [&](size_t stride, size_t len)
            {
                return (i / stride) % len;
            });
            mlopen::unpack(f, indices);
        });
    }
};

template<class... Ts>
auto par_ford(Ts... xs) MLOPEN_RETURNS
(
    std::bind(par_ford_impl{}, std::placeholders::_1, xs...)
);

#endif
