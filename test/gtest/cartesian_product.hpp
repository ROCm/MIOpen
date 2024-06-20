/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <tuple>
#include <vector>

namespace test {

// converts a vector of a type to a vector of tuples of the type
template <typename T>
inline std::vector<std::tuple<T>> vec_to_tuple(std::vector<T> in)
{
    std::vector<std::tuple<T>> out{};

    for(auto i : in)
    {
        out.push_back(std::make_tuple(i));
    }

    return out;
}

template <typename T, typename... Ts>
using SameType = std::enable_if_t<std::conjunction_v<std::is_same<T, Ts>...>>;

// produces the cartesian product of the tuples in A and two copies of the elements of B
// all elements must be the same type
template <typename T, typename... Ts, typename = SameType<T, Ts...>>
inline auto cartesian_product_abb(const std::vector<std::tuple<Ts...>>& A, const std::vector<T>& B)
    -> std::vector<decltype(std::tuple_cat(A[0], std::make_tuple(B[0], B[0])))>
{
    auto C = std::vector<decltype(std::tuple_cat(A[0], std::make_tuple(B[0], B[0])))>{};
    for(auto a : A)
    {
        for(auto b : B)
        {
            for(auto b2 : B)
            {
                C.push_back(std::tuple_cat(a, std::make_tuple(b, b2)));
            }
        }
    }

    return C;
}

// produces the cartesian product of the elements of A and two copies of the elements of B
template <typename T>
inline auto cartesian_product_abb(const std::vector<T>& A, const std::vector<T>& B)
    -> std::vector<std::tuple<T, T, T>>
{
    return cartesian_product_abb(vec_to_tuple(A), B);
}

// produces the cartesian product of the elements of A and two copies each
// of the elements of B and C
template <typename T>
inline auto cartesian_product_axx(const std::vector<T>& A,
                                  const std::vector<T>& B,
                                  const std::vector<T>& C) -> std::vector<std::tuple<T, T, T, T, T>>
{
    auto product = cartesian_product_abb(A, B);
    return cartesian_product_abb(product, C);
}

// produces the cartesian product of the elements of A and two copies each
// of the elements of B, C and D
template <typename T>
inline auto cartesian_product_axx(const std::vector<T>& A,
                                  const std::vector<T>& B,
                                  const std::vector<T>& C,
                                  const std::vector<T>& D)
    -> std::vector<std::tuple<T, T, T, T, T, T, T>>
{
    auto product  = cartesian_product_abb(A, B);
    auto product2 = cartesian_product_abb(product, C);
    return cartesian_product_abb(product2, D);
}

} // namespace test
