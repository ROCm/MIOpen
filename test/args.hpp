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

#ifndef GUARD_MIOPEN_TEST_ARGS_HPP
#define GUARD_MIOPEN_TEST_ARGS_HPP

#include <algorithm>
#include <cassert>
#include <functional>

#include <miopen/logger.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <miopen/rank.hpp>
#include <miopen/type_name.hpp>

namespace args {

using string_map = std::unordered_map<std::string, std::vector<std::string>>;

template <class IsKeyword>
string_map parse(std::vector<std::string> as, IsKeyword is_keyword)
{
    string_map result;

    std::string flag;
    for(auto&& x : as)
    {
        if(is_keyword(x))
        {
            flag = x;
            result[flag]; // Ensure the flag exists
        }
        else
        {
            result[flag].push_back(x);
        }
    }
    return result;
}

namespace detail {

template <class T>
auto is_container(miopen::rank<1>, T&& x)
    -> decltype(x.insert(x.end(), *x.begin()), std::true_type{});

template <class T>
std::false_type is_container(miopen::rank<0>, T&&);

template <class T, class U>
auto is_input_streamable(miopen::rank<1>, T&& x, U&& y) -> decltype((x >> y), std::true_type{});

template <class T, class U>
std::false_type is_input_streamable(miopen::rank<0>, T&&, U&&);

template <class T, class U>
auto is_output_streamable(miopen::rank<1>, T&& x, U&& y) -> decltype((x << y), std::true_type{});

template <class T, class U>
std::false_type is_output_streamable(miopen::rank<0>, T&&, U&&);

template <bool B>
struct requires_bool
{
    static const bool value = B;
};

template <class T, int64_t N>
struct requires_unwrap : T
{
};
} // namespace detail

template <class T>
struct is_container : decltype(detail::is_container(miopen::rank<1>{}, std::declval<T>()))
{
};

template <class T>
struct is_input_streamable : decltype(detail::is_input_streamable(
                                 miopen::rank<1>{},
                                 std::declval<std::istream>(),
                                 std::declval<typename std::add_lvalue_reference<T>::type>()))
{
};

template <class T>
struct is_output_streamable : decltype(detail::is_output_streamable(
                                  miopen::rank<1>{},
                                  std::declval<std::ostream>(),
                                  std::declval<typename std::add_lvalue_reference<T>::type>()))
{
};

#ifdef _MSC_VER
#define ARGS_REQUIRES_BOOL(...)                                                           \
    args::detail::requires_unwrap<decltype(args::detail::requires_bool<(__VA_ARGS__)>{}), \
                                  __LINE__>::value
#else
#define ARGS_REQUIRES_BOOL(...) (__VA_ARGS__)
#endif

#define ARGS_REQUIRES(...)                                                                  \
    bool MIOPEN_PP_CAT(                                                                     \
        RequiresBool,                                                                       \
        __LINE__)                          = true,                                          \
        typename std::enable_if<ARGS_REQUIRES_BOOL(MIOPEN_PP_CAT(RequiresBool, __LINE__) && \
                                                   (__VA_ARGS__)),                          \
                                int>::type = 0

template <class T>
struct value_parser
{
    template <ARGS_REQUIRES(is_input_streamable<T>{} and not std::is_pointer<T>{} and
                            not std::is_enum<T>{})>
    static T apply(const std::string& x)
    {
        T result;
        std::stringstream ss;
        ss.str(x);
        ss >> result;
        if(ss.fail())
            throw std::runtime_error("Failed to parse: " + x);
        return result;
    }

    template <ARGS_REQUIRES(std::is_enum<T>{})>
    static T apply(const std::string& x)
    {
        std::ptrdiff_t i;
        std::stringstream ss;
        ss.str(x);
        ss >> i;
        if(ss.fail())
            throw std::runtime_error("Failed to parse: " + x);
        return static_cast<T>(i);
    }
};

struct any_value
{
    std::string s;

    template <class T,
              class = decltype(value_parser<T>::apply(std::string{})),
              ARGS_REQUIRES(not std::is_enum<T>{})>
    operator T() const
    {
        return value_parser<T>::apply(s);
    }
};

template <class T, std::size_t... Ns, class Data>
auto any_construct_impl(miopen::rank<1>, std::index_sequence<Ns...>, const Data& d)
    -> decltype(T(any_value{d[Ns]}...))
{
    return T(any_value{d[Ns]}...);
}

template <class T, std::size_t... Ns, class Data>
T any_construct_impl(miopen::rank<0>, std::index_sequence<Ns...>, const Data&)
{
    throw std::runtime_error("Cannot construct: " + miopen::get_type_name<T>());
}

template <class T, std::size_t N, class Data>
T any_construct(const Data& d)
{
    return any_construct_impl<T>(miopen::rank<1>{}, std::make_index_sequence<N>(), d);
}

struct write_value
{
    template <class T>
    using is_multi_value =
        std::integral_constant<bool,
                               (is_container<T>{} and not std::is_convertible<T, std::string>{})>;

    template <class Container, ARGS_REQUIRES(is_multi_value<Container>{})>
    void operator()(Container& result, std::vector<std::string> params) const
    {
        using value_type = typename Container::value_type;
        for(auto&& s : params)
        {
            result.insert(result.end(), value_parser<value_type>::apply(s));
        }
    }

    template <class T, ARGS_REQUIRES(not is_multi_value<T>{} and is_input_streamable<T>{})>
    void operator()(T& result, std::vector<std::string> params) const
    {
        if(params.empty())
            throw std::runtime_error("Missing argument");
        result = value_parser<T>::apply(params.back());
    }

    template <class T, ARGS_REQUIRES(not is_multi_value<T>{} and not is_input_streamable<T>{})>
    void operator()(T& result, std::vector<std::string> params) const
    {
        switch(params.size())
        {
        case 0: {
            result = any_construct<T, 0>(params);
            break;
        }
        case 1: {
            result = any_construct<T, 1>(params);
            break;
        }
        case 2: {
            result = any_construct<T, 2>(params);
            break;
        }
        case 3: {
            result = any_construct<T, 3>(params);
            break;
        }
        case 4: {
            result = any_construct<T, 4>(params);
            break;
        }
        case 5: {
            result = any_construct<T, 5>(params);
            break;
        }
        case 6: {
            result = any_construct<T, 6>(params);
            break;
        }
        case 7: {
            result = any_construct<T, 7>(params);
            break;
        }
        default: throw std::runtime_error("Cannot construct: " + miopen::get_type_name<T>());
        }
    }
};

struct read_value
{
    template <class Container,
              ARGS_REQUIRES(is_container<Container>{} and not is_output_streamable<Container>{})>
    std::string operator()(Container& xs) const
    {
        if(xs.begin() == xs.end())
            return "";
        return std::accumulate(xs.begin() + 1, xs.end(), (*this)(*xs.begin()), [&](auto x, auto y) {
            return x + " " + (*this)(y);
        });
    }

    template <class T, ARGS_REQUIRES(is_output_streamable<T>{})>
    std::string operator()(T& x) const
    {
        std::stringstream ss;
        ss << x;
        return ss.str();
    }
};

} // namespace args

#endif // GUARD_MIOPEN_TEST_ARGS_HPP
