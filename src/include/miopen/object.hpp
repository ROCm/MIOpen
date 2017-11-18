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
#ifndef GUARD_MIOPEN_HANDLE_HPP
#define GUARD_MIOPEN_HANDLE_HPP

#include <miopen/rank.hpp>

#if defined(MIOPEN_USE_CLANG_TIDY)
#define MIOPEN_OBJECT_CAST reinterpret_cast
#else
#define MIOPEN_OBJECT_CAST static_cast
#endif

#define MIOPEN_DEFINE_OBJECT(object, ...)                                \
    inline __VA_ARGS__& miopen_get_object(object& obj)                   \
    {                                                                    \
        return MIOPEN_OBJECT_CAST<__VA_ARGS__&>(obj);                    \
    }                                                                    \
    inline const __VA_ARGS__& miopen_get_object(const object& obj)       \
    {                                                                    \
        return MIOPEN_OBJECT_CAST<const __VA_ARGS__&>(obj);              \
    }                                                                    \
    inline void miopen_destroy_object(object* p)                         \
    {                                                                    \
        miopen::detail::delete_obj(MIOPEN_OBJECT_CAST<__VA_ARGS__*>(p)); \
    }

namespace miopen {

namespace detail {

template <class T>
void delete_obj(T* x)
{
    delete x; // NOLINT
}

template <class T>
T& get_object_impl(rank<0>, T& x)
{
    return x;
}

template <class T>
auto get_object_impl(rank<1>, T& x) -> decltype(miopen_get_object(x))
{
    return miopen_get_object(x);
}

} // namespace detail

template <class T>
auto get_object(T& x) -> decltype(detail::get_object_impl(rank<1>{}, x))
{
    return detail::get_object_impl(rank<1>{}, x);
}

} // namespace miopen

#endif
