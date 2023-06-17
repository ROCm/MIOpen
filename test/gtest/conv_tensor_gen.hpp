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

#include <gtest/gtest.h>

#include <random>

// Copied from conv_driver.hpp

template <typename T>
inline T FRAND()
{
    double d = static_cast<double>(rand() / (static_cast<double>(RAND_MAX)));
    return static_cast<T>(d);
}

template <typename T>
inline T RAN_GEN(T A, T B)
{
    T r = (FRAND<T>() * (B - A)) + A;
    return r;
}
template <typename T>
T RanGenData()
{
    return RAN_GEN<T>(static_cast<T>(0.0f), static_cast<T>(1.0f));
}

template <>
float8 RanGenData()
{
    return RAN_GEN<float8>(static_cast<float8>(-1.0f), static_cast<float8>(1.0f));
}

template <>
bfloat8 RanGenData()
{
    const auto tmp = RAN_GEN<float>(static_cast<float>(-1.0f), static_cast<float>(1.0f));
    return static_cast<bfloat8>(tmp);
}

template <typename T>
struct GenData
{
    template <class... Ts>
    T operator()(Ts...) const
    {
        return RanGenData<T>();
    }
};

template <typename T>
T RanGenWeights()
{
    return RAN_GEN<T>(static_cast<T>(-0.5), static_cast<T>(0.5));
}

// Shift FP16 distribution towards positive numbers,
// otherwise Winograd FP16 validation fails.
template <>
half_float::half RanGenWeights()
{
    return RAN_GEN<half_float::half>(static_cast<half_float::half>(-1.0 / 3.0),
                                     static_cast<half_float::half>(0.5));
}

template <>
float8 RanGenWeights()
{
    const auto tmp =
        RAN_GEN<float>(0.0, 1.0) > 0.5 ? static_cast<float>(0.0) : static_cast<float>(1.0);
    // 1 in 2 chance of number being positive
    const float sign =
        (RAN_GEN<float>(0.0, 1.0) > 0.5) ? static_cast<float>(-1) : static_cast<float>(1);
    const auto tmp2 = static_cast<float>(std::numeric_limits<float8>::epsilon()) *
                      static_cast<float>(2) * sign * static_cast<float>(tmp);
    return static_cast<float8>(tmp2);
}

template <>
bfloat8 RanGenWeights()
{
    const auto tmp =
        RAN_GEN<float>(0.0, 1.0) > 0.5 ? static_cast<float>(0.0) : static_cast<float>(1.0);
    // 1 in 2 chance of number being positive
    const float sign =
        (RAN_GEN<float>(0.0, 1.0) > 0.5) ? static_cast<float>(-1) : static_cast<float>(1);
    const auto tmp2 = static_cast<float>(std::numeric_limits<float8>::epsilon()) *
                      static_cast<float>(2) * sign * static_cast<float>(tmp);
    return static_cast<bfloat8>(tmp2);
}

template <typename T>
struct GenWeights
{
    template <class... Ts>
    T operator()(Ts...) const
    {
        return RanGenWeights<T>();
    }
};
