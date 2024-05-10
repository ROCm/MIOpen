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
#include <hip_float8.hpp>

#include "../random.hpp"

template <typename T>
inline T RanGenData()
{
    return prng::gen_canonical<T>();
}

template <>
inline float8 RanGenData()
{
    return prng::gen_A_to_B(static_cast<float8>(-1.0f), static_cast<float8>(1.0f));
}

template <>
inline bfloat8 RanGenData()
{
    return prng::gen_A_to_B(static_cast<bfloat8>(-1.0f), static_cast<bfloat8>(1.0f));
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
    return prng::gen_A_to_B(static_cast<T>(-0.5), static_cast<T>(0.5));
}

// Shift FP16 distribution towards positive numbers,
// otherwise Winograd FP16 validation fails.
template <>
inline half_float::half RanGenWeights()
{
    return prng::gen_A_to_B(static_cast<half_float::half>(-1.0 / 3.0),
                            static_cast<half_float::half>(0.5));
}

template <>
inline float8 RanGenWeights()
{
    const auto tmp = prng::gen_canonical<float>() > 0.5f ? 0.0f : 2.0f;
    // 1 in 2 chance of number being positive
    const auto sign = prng::gen_canonical<float>() > 0.5f ? -1.0f : 1.0f;
    return static_cast<float8>(static_cast<float>(std::numeric_limits<float8>::epsilon()) * sign *
                               tmp);
}

template <>
inline bfloat8 RanGenWeights()
{
    const auto tmp = prng::gen_canonical<float>() > 0.5f ? 0.0f : 2.0f;
    // 1 in 2 chance of number being positive
    const auto sign = prng::gen_canonical<float>() > 0.5f ? -1.0f : 1.0f;
    return static_cast<bfloat8>(static_cast<float>(std::numeric_limits<bfloat8>::epsilon()) * sign *
                                tmp);
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
