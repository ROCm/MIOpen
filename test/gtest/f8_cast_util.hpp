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

#include <hip_float8.hpp>
#include "verify.hpp"
using float8_fnuz  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8_fnuz = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;

template <typename U, typename V>
struct Fp8Cast
{
    uint64_t seed = 1234;
    bool is_stoch = true;
    V operator()(U x)
    {
        if(is_stoch)
        {
            auto tmp = float8_fnuz(
                static_cast<float>(x), miopen_f8::hip_f8_rounding_mode::stochastic, seed);
            return static_cast<V>(tmp);
        }
        else
        {
            auto tmp = float8_fnuz(static_cast<float>(x));
            return static_cast<V>(tmp);
        }
    }
};

template <typename U, typename V>
struct Bf8Cast
{
    uint64_t seed = 1234;
    bool is_stoch = true;
    V operator()(U x)
    {
        if(is_stoch)
        {
            auto tmp = bfloat8_fnuz(
                static_cast<float>(x), miopen_f8::hip_f8_rounding_mode::stochastic, seed);
            return static_cast<V>(tmp);
        }
        else
        {
            auto tmp = bfloat8_fnuz(static_cast<float>(x));
            return static_cast<V>(tmp);
        }
    }
};
