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
#ifndef GUARD_DRIVER_CONV_COMMON_HPP
#define GUARD_DRIVER_CONV_COMMON_HPP

/// This module is introduced in order to get rid of the followinf false linker warnings:
/// ld.lld: error: duplicate symbol: signed char detail::RanGenWeights<signed char>()
/// >>> defined at conv1.cpp
/// >>>            CMakeFiles/MIOpenDriver.dir/conv1.cpp.o:(signed char detail::RanGenWeights<signed
/// char>())
/// >>> defined at conv3.cpp
/// >>>            CMakeFiles/MIOpenDriver.dir/conv3.cpp.o:(.text+0x10)

#include "random.hpp"

#include <miopen/bfloat16.hpp>

#include <cstdint>
#if !defined(_WIN32)
#include <half/half.hpp>
#else
#include <half.hpp>
#endif
using half         = half_float::half;
using hip_bfloat16 = bfloat16;
#include <hip_float8.hpp>

using float16 = half_float::half;
using float8  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8 = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;

namespace conv {

template <typename T>
T RanGenWeights()
{
    return prng::gen_A_to_B(static_cast<T>(-0.5), static_cast<T>(0.5));
}

template <>
float16 RanGenWeights();
template <>
int8_t RanGenWeights();
template <>
float8 RanGenWeights();
template <>
bfloat8 RanGenWeights();

} // namespace conv

#endif // GUARD_DRIVER_CONV_COMMON_HPP
