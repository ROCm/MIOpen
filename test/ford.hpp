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
#ifndef GUARD_FORD_HPP
#define GUARD_FORD_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <miopen/par_for.hpp>
#include <miopen/each_args.hpp>
#include <miopen/returns.hpp>
#include <numeric>
#include <vector>

#ifdef __MINGW32__
#include <mingw.thread.h>
#else
#include <thread>
#endif

#include <future>

// An improved async, that doesn't block
template <class Function>
std::future<typename std::result_of<Function()>::type> detach_async(Function&& f)
{
    using result_type = typename std::result_of<Function()>::type;
    std::packaged_task<result_type()> task(std::forward<Function>(f));
    auto fut = task.get_future();
    std::thread(std::move(task)).detach();
    return std::move(fut);
}

template <class T, class Work>
auto then(std::future<T> f, Work w) -> std::future<decltype(w(f.get()))>
{
    return std::async(std::launch::deferred,
                      [ =, f = std::move(f) ]() mutable { return w(f.get()); });
}

struct joinable_thread : std::thread
{
    template <class... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...) // NOLINT
    {
    }

    joinable_thread& operator=(joinable_thread&& other) = default;
    joinable_thread(joinable_thread&& other)            = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

struct thread_factory
{
    template <class F>
    joinable_thread operator()(std::size_t& work, std::size_t n, std::size_t grainsize, F f) const
    {
        auto result = joinable_thread([=] {
            std::size_t start = work;
            std::size_t last  = std::min(n, work + grainsize);
            for(std::size_t i = start; i < last; i++)
            {
                f(i);
            }
        });
        work += grainsize;
        return result;
    }
};

using miopen::par_for; // NOLINT

template <class T>
struct ford_wrapper
{
    template <class... Ts>
    auto operator()(Ts... xs) const MIOPEN_RETURNS(std::bind(T{}, std::placeholders::_1, xs...));
};

// Multidimensional for loop
struct ford_impl
{
    template <class F>
    void operator()(F f) const
    {
        f();
    }

    template <class F, class T, class... Ts>
    void operator()(F f, T x, Ts... xs) const
    {
        // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=55914
        for(T i = 0; i < x; i++)
        {
            (*this)([&](Ts... is) { f(i, is...); }, xs...);
        }
    }
};

static constexpr ford_wrapper<ford_impl> ford{};

struct par_ford_impl
{
    template <class F, class... Ts>
    void operator()(F f, Ts... xs) const
    {
        using array_type = std::array<std::size_t, sizeof...(Ts)>;
        array_type lens  = {{static_cast<std::size_t>(xs)...}};
        array_type strides;
        strides.fill(1);
        std::partial_sum(
            lens.rbegin(), lens.rend() - 1, strides.rbegin() + 1, std::multiplies<std::size_t>());
        auto size = std::accumulate(lens.begin(), lens.end(), 1, std::multiplies<std::size_t>());
        par_for(size, [&](std::size_t i) {
            array_type indices;
            std::transform(strides.begin(),
                           strides.end(),
                           lens.begin(),
                           indices.begin(),
                           [&](size_t stride, size_t len) { return (i / stride) % len; });
            miopen::unpack(f, indices);
        });
    }
};

static constexpr ford_wrapper<par_ford_impl> par_ford{};

#endif
