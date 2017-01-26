#ifndef GUARD_FORD_HPP
#define GUARD_FORD_HPP

#include <mlopen/returns.hpp>
#include <mlopen/each_args.hpp>
#include <functional>
#include <algorithm>
#include <numeric>
#include <thread>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>

struct joinable_thread : std::thread
{
    template<class... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {}

    joinable_thread& operator=( joinable_thread&& other )=default;
    joinable_thread( joinable_thread&& other )=default;

    ~joinable_thread()
    {
        if (this->joinable()) this->join();
    }
};

struct thread_factory
{
    template<class F>
    joinable_thread operator()(std::size_t& work, std::size_t n, std::size_t grainsize, F f)
    {
        auto result = joinable_thread([&, work]
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
    }
};

template<class F>
void par_for(std::size_t n, std::size_t threadsize, F f)
{
    if (threadsize <= 1)
    {
        for(std::size_t i=0;i<n;i++) f(i);
    }
    else
    {
        std::vector<joinable_thread> threads(threadsize);
        const std::size_t grainsize = std::ceil(static_cast<double>(n) / threads.size());

        std::size_t work = 0;
        std::generate(threads.begin(), threads.end(), std::bind(thread_factory{}, std::ref(work), n, grainsize, f));
        assert(work >= n);
    }
}

template<class F>
void par_for(std::size_t n, F f)
{
    const int min_grain = 8;
    const auto threadsize = std::min<std::size_t>(std::thread::hardware_concurrency(), n/min_grain);
    par_for(n, threadsize, f);
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
