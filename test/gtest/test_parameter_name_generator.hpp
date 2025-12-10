/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <algorithm>
#include <concepts>
#include <ostream>
#include <ranges>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "gtest_common.hpp"

namespace {

// Concept for printable objets which support the << operator.
template <typename T>
concept Printable = requires(std::ostream& os, T t)
{
    os << t;
};

// Concept for printable containers, elements of which support the << operator.
template <typename T>
concept PrintableElement = requires(std::ostream& os, T t)
{
    os << t[0];
};

// Concept for iterable containers.
template <typename T>
concept Container = requires(T t)
{
    // clang-format off
    {
        std::size(t)
    } -> std::same_as<std::size_t>;
    {
        std::begin(t)
    } -> std::same_as<typename T::iterator>;
    {
        std::end(t)
    } -> std::same_as<typename T::iterator>;
    // clang-format on
};

// Template wrapper around a test parameter.
// It defines the << operator as required by GTest, and the cast operator that returns the wrapped
// parameter. If you need to wrap a container, use the 'NamedContainer' template wrapper.
template <typename T>
requires Printable<T> && std::is_move_constructible_v<T>
struct NamedParameter
{
    NamedParameter(std::string parameterName, T parameterValue) noexcept
        : name(std::move(parameterName)), value(std::move(parameterValue))
    {
    }

    operator T() const { return value; }

    T& operator()() noexcept { return value; }

    const T& operator()() const noexcept { return value; }

    friend std::ostream& operator<<(std::ostream& os, const NamedParameter<T>& param)
    {
        return os << param.name << ": " << param.value;
    }

    std::string name{};
    T value{};
};

// Template wrapper around an iterable container test parameter (like all STL containers).
// It defines the << operator as required by GTest, and the cast operator that returns the wrapped
// container. If you need to wrap a parameter which is not a container, use the 'NamedParameter'
// template wrapper.
template <typename T>
requires Container<T> && PrintableElement<T> && std::is_move_constructible_v<T>
struct NamedContainer
{
    NamedContainer(std::string containerName,
                   T containerValue,
                   std::string valueSeparator = " ") noexcept
        : name(std::move(containerName)),
          value(std::move(containerValue)),
          separator(std::move(valueSeparator))
    {
    }

    operator T() const { return value; }

    T& operator()() noexcept { return value; }

    const T& operator()() const noexcept { return value; }

    friend std::ostream& operator<<(std::ostream& os, const NamedContainer<T>& param)
    {
        os << param.name << ": [";

        if(param.value.size() > 0)
        {
            os << *param.value.begin();

            for(auto it = param.value.begin() + 1; it != param.value.end(); ++it)
            {
                os << param.separator << *it;
            }
        }

        os << "]";

        return os;
    }

    std::string name{};
    T value{};
    std::string separator{};
};

// Variadic template function that creates a GTest ValueArray of 'NamedParameter' each one with the
// name and value supplied. The result can be directly fed to the GTest instantiated test suite.
//
// Example:
//
//      testing::Combine(
//          MakeNamedParameterValues<int>("TestInt1", 1, 2, 3),
//          MakeNamedParameterValues<int>("TestInt2", 10, 20, 30),
//          MakeNamedParameterValues<int>("TestInt3", 100, 200, 300)
//      );
//
template <typename... T>
static auto MakeNamedParameterValues(const std::string& name, T... values)
{
    return testing::Values(NamedParameter<T>{name, values}...);
}

// Variadic template function that creates a GTest ValueArray of 'NamedContainer' each one with the
// name and value supplied. An optional separator for each value in each containers can be supplied.
// The result can be directly fed to the GTest instantiated test suite.
//
// Example:
//
//      std::set<std::vector<int>> tensorSizes = ...
//
//      testing::Combine(
//          MakeNamedParameterCollectionValues<std::vector<int>>("TestTensorSizes", tensorSizes,
//          "x"), MakeNamedParameterValues<int>("TestInteger", 1, 2, 3)
//      );
//
// The 'tensorSizes' collection of std::vector<int>'s is turned into a collection of
// NamedContainer<std::vector<int>>, and then fed into 'testing::Combine()'.
//
template <typename T>
static auto MakeNamedParameterCollectionValues(const std::string& name,
                                               const std::ranges::range auto& collection,
                                               std::string separator = " ")
{
    std::vector<NamedContainer<T>> v;

    v.reserve(collection.size());

    for(const auto& x : collection)
    {
        v.emplace_back(name, x, separator);
    }

    return testing::ValuesIn(v);
}

// The 'GetRangeAsString()' function Returns the string representation of the input collection,
// using the user-supplied separator. The returned string meets the GTest requirements for test
// names: only alphanumeric characters and underscores are allowed. Any dot character (i.e. '.') is
// turned into a 'p' character, which stands for 'point'.
//
// Examples:
//
//      GetRangeAsString(std::vector<int>{1, 2, 3, 4}, "x") returns "1x2x3x4"
//      GetRangeAsString(std::vector<float>{1.1, 2.2, 3.3, 4.4}, ", ") returns "1p1, 2p2, 3p3, 4p4"
//
static std::string GetRangeAsString(const std::ranges::range auto& r,
                                    std::string_view separator = " ")
{
    std::string str;

    if(r.size() > 0)
    {
        std::stringstream ss;

        ss << *r.begin();

        for(auto it = r.begin() + 1; it != r.end(); ++it)
        {
            ss << separator << *it;
        }

        str = ss.str();

        // Name format only supports letters, numbers and underscores.
        std::transform(
            str.begin(), str.end(), str.begin(), [](char c) { return (c == '.') ? 'p' : c; });
    }

    return str;
}

} // namespace
