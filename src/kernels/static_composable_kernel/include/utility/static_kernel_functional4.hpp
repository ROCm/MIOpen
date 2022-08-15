/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef CK_FUNCTIONAL4_HPP
#define CK_FUNCTIONAL4_HPP

#include "static_kernel_sequence.hpp"
#include "static_kernel_tuple.hpp"
#include "static_kernel_array.hpp"

namespace ck {

namespace detail {

template <typename Indices>
struct unpack_impl;

template <index_t... Is>
struct unpack_impl<Sequence<Is...>>
{
    template <typename F, typename X>
    __host__ __device__ constexpr auto operator()(F f, const X& x) const
    {
        return f(x.At(Number<Is>{})...);
    }
};

} // namespace detail

template <typename F, typename X>
__host__ __device__ constexpr auto unpack(F f, const X& x)
{
    return detail::unpack_impl<typename arithmetic_sequence_gen<0, X::Size(), 1>::type>{}(f, x);
}

} // namespace ck
#endif
