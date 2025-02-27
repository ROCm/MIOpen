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
#ifndef GUARD_MIOPEN_TEST_RANDOM_HPP
#define GUARD_MIOPEN_TEST_RANDOM_HPP

#include "../driver/random.hpp"

namespace prng {
template <typename T>
inline T gen_descreet_uniform_sign(double scale, int32_t range)
{
    return static_cast<T>(scale * prng::gen_A_to_B(-range + 1, range));
}

template <typename T>
inline T gen_descreet_unsigned(double scale, int32_t range)
{
    return static_cast<T>(scale * static_cast<double>(gen_0_to_B(range)));
}

} // namespace prng

// lambda factory
template <typename T, typename ScaleT, typename RangeT>
auto uniform_signed_initializer(ScaleT scale_arg, RangeT range_arg)
{
    return [=](auto&&...) -> T {
        // uniform sign give balance of both negative and positive values
        return prng::gen_descreet_uniform_sign<T>(scale_arg, range_arg);
    };
}

template <typename T, typename ScaleT, typename RangeT>
auto uniform_unsigned_initializer(ScaleT scale_arg, RangeT range_arg)
{
    return [=](auto&&...) -> T { return prng::gen_descreet_unsigned<T>(scale_arg, range_arg); };
}

#endif // GUARD_MIOPEN_TEST_RANDOM_HPP
