/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_GUARD_MLOPEN_PAR_FOR_HPP
#define MIOPEN_GUARD_MLOPEN_PAR_FOR_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#ifdef __MINGW32__
#include <mingw.thread.h>
#else
#include <thread>
#endif

namespace miopen {

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

template <class F>
void par_for_impl(std::size_t n, std::size_t threadsize, F f)
{
    if(threadsize <= 1)
    {
        for(std::size_t i = 0; i < n; i++)
            f(i);
    }
    else
    {
        std::vector<joinable_thread> threads(threadsize);
        const std::size_t grainsize = std::ceil(static_cast<double>(n) / threads.size());

        std::size_t work = 0;
        std::generate(threads.begin(),
                      threads.end(),
                      std::bind(thread_factory{}, std::ref(work), n, grainsize, f));
        assert(work >= n);
    }
}

template <class F>
void par_for(std::size_t n, std::size_t min_grain, F f)
{
    const auto threadsize =
        std::min<std::size_t>(std::thread::hardware_concurrency(), n / min_grain);
    par_for_impl(n, threadsize, f);
}

struct min_grain
{
    std::size_t n = 0;
};

template <class F>
void par_for(std::size_t n, min_grain mg, F f)
{
    const auto threadsize = std::min<std::size_t>(std::thread::hardware_concurrency(), n / mg.n);
    par_for_impl(n, threadsize, f);
}

template <class F>
void par_for(std::size_t n, F f)
{
    par_for(n, min_grain{8}, f);
}

struct max_threads
{
    std::size_t n = 0;
};

template <class F>
void par_for(std::size_t n, max_threads mt, F f)
{
    const auto threadsize = std::min<std::size_t>(std::thread::hardware_concurrency(), mt.n);
    par_for_impl(n, std::min(threadsize, n), f);
}

template <class F>
void par_for_strided(std::size_t n, max_threads mt, F f)
{
    auto threadsize = std::min<std::size_t>(std::thread::hardware_concurrency(), mt.n);
    par_for_impl(threadsize, threadsize, [&](auto start) {
        for(std::size_t i = start; i < n; i += threadsize)
        {
            f(i);
        }
    });
}

} // namespace miopen

#endif
