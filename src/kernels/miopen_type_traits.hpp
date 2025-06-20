/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#pragma once

#if HIP_PACKAGE_VERSION_FLAT < 7000000000ULL
#ifdef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS

namespace std {

template <class T>
struct remove_reference
{
    typedef T type;
};
template <class T>
struct remove_reference<T&>
{
    typedef T type;
};
template <class T>
struct remove_reference<T&&>
{
    typedef T type;
};

template <class T>
using remove_reference_t = typename remove_reference<T>::type;

template <class T>
struct remove_const
{
    typedef T type;
};
template <class T>
struct remove_const<const T>
{
    typedef T type;
};

template <class T>
struct remove_volatile
{
    typedef T type;
};
template <class T>
struct remove_volatile<volatile T>
{
    typedef T type;
};

template <class T>
struct remove_cv
{
    typedef typename remove_volatile<typename remove_const<T>::type>::type type;
};

#if HIP_PACKAGE_VERSION_FLAT >= 6000025000ULL && HIP_PACKAGE_VERSION_FLAT < 6001024000ULL
template <class T, T v>
struct integral_constant
{
    static constexpr T value = v;
    using value_type         = T;
    using type               = integral_constant;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

using true_type  = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <class T, class U>
struct is_same : false_type
{
};

template <class T>
struct is_same<T, T> : true_type
{
};

template <bool B, typename T = void>
using enable_if = __hip_internal::enable_if<B, T>;

template <bool B, typename T = void>
using enable_if_t = typename __hip_internal::enable_if<B, T>::type;
#endif

template <class T>
struct is_pointer_helper : false_type
{
};

template <class T>
struct is_pointer_helper<T*> : true_type
{
};

template <class T>
struct is_pointer : is_pointer_helper<typename remove_cv<T>::type>
{
};

template <bool predicate, typename X, typename Y>
struct conditional;

template <typename X, typename Y>
struct conditional<true, X, Y>
{
    using type = X;
};

template <typename X, typename Y>
struct conditional<false, X, Y>
{
    using type = Y;
};

template <bool predicate, typename X, typename Y>
using conditional_t = typename conditional<predicate, X, Y>::type;

} // namespace std
#else

#include <type_traits> // std::remove_reference, std::remove_cv, is_pointer

#endif
#else
#include <type_traits>
#endif
