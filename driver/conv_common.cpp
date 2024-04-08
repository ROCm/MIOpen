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
#include "conv_common.hpp"

namespace conv {

// Shift FP16 distribution towards positive numbers,
// otherwise Winograd FP16 validation fails.
template <>
float16 RanGenWeights()
{
    return prng::gen_A_to_B(static_cast<float16>(-1.0 / 3.0), static_cast<float16>(0.5));
}

// int8 has it's own range
template <>
int8_t RanGenWeights()
{
    return prng::gen_A_to_B(static_cast<int8_t>(-1), static_cast<int8_t>(1));
}

template <>
float8 RanGenWeights()
{
    const auto tmp =
        prng::gen_0_to_B(1.0) > 0.5 ? static_cast<float>(0.0) : static_cast<float>(1.0);
    // 1 in 2 chance of number being positive
    const float sign =
        (prng::gen_0_to_B(1.0) > 0.5) ? static_cast<float>(-1) : static_cast<float>(1);
    const auto tmp2 = static_cast<float>(std::numeric_limits<float8>::epsilon()) *
                      static_cast<float>(2) * sign * static_cast<float>(tmp);
    return static_cast<float8>(tmp2);
}

template <>
bfloat8 RanGenWeights()
{
    const auto tmp =
        prng::gen_0_to_B(1.0) > 0.5 ? static_cast<float>(0.0) : static_cast<float>(1.0);
    // 1 in 2 chance of number being positive
    const float sign =
        (prng::gen_0_to_B(1.0) > 0.5) ? static_cast<float>(-1) : static_cast<float>(1);
    const auto tmp2 = static_cast<float>(std::numeric_limits<float8>::epsilon()) *
                      static_cast<float>(2) * sign * static_cast<float>(tmp);
    return static_cast<bfloat8>(tmp2);
}

} // namespace conv
