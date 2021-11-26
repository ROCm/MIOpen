/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef HOST_TENSOR_GENERATOR_HPP
#define HOST_TENSOR_GENERATOR_HPP

#include <cmath>
#include "config.hpp"

struct GeneratorTensor_1
{
    int value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        return value;
    }
};

struct GeneratorTensor_0
{
    int value = 0;

    template <typename... Is>
    float operator()(Is...)
    {
        return value;
    }
};

struct GeneratorTensor_2
{
    int min_value = 0;
    int max_value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        return (std::rand() % (max_value - min_value)) + min_value;
    }
};

template <typename T>
struct GeneratorTensor_3
{
    T min_value = 0;
    T max_value = 1;

    template <typename... Is>
    float operator()(Is...)
    {
        float tmp = float(std::rand()) / float(RAND_MAX);

        return min_value + tmp * (max_value - min_value);
    }
};

struct GeneratorTensor_Checkboard
{
    template <typename... Ts>
    float operator()(Ts... Xs) const
    {
        std::array<ck::index_t, sizeof...(Ts)> dims = {{static_cast<ck::index_t>(Xs)...}};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](bool init, ck::index_t x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};

#endif
