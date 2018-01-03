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
#ifndef GUARD_GEMM_HPP
#define GUARD_GEMM_HPP

#include "ford.hpp"
#include <miopen/returns.hpp>

template <class AF, class BF, class CF>
void gemm(std::size_t n, std::size_t m, std::size_t k, AF a, BF b, CF c)
{
    auto inner_loop = [&](int i, int j) {
        double x = 0.0;
        ford(k)([&](int kk) { x += a(i, kk) * b(kk, j); });
        c(i, j, x);
    };
    if(n * m > 32)
    {
        par_ford(n, m)(inner_loop);
    }
    else
    {
        ford(n, m)(inner_loop);
    }
}

struct with_stride_impl
{
    template <class T>
    auto operator()(T& data, std::size_t stride, std::size_t x, std::size_t y) const
        MIOPEN_RETURNS(data[x * stride + y]);

    template <class T>
    auto operator()(std::vector<T>& data, std::size_t stride, std::size_t x, std::size_t y) const
        MIOPEN_RETURNS(data.at(x* stride + y));
};

template <class T>
auto with_stride(T& data, std::size_t stride) MIOPEN_RETURNS(std::bind(
    with_stride_impl{}, std::ref(data), stride, std::placeholders::_1, std::placeholders::_2));

template <class T>
auto with_stride(T* data, std::size_t stride) MIOPEN_RETURNS(
    std::bind(with_stride_impl{}, data, stride, std::placeholders::_1, std::placeholders::_2));

#endif
