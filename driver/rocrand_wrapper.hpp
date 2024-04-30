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
#ifndef GUARD_ROCRAND_WRAPPER_HPP
#define GUARD_ROCRAND_WRAPPER_HPP

// The wrapper is necessary because direct inclusion of rocrand.hpp into the driver source leads to
// build errors like this: "amd_hip_fp16.h:1743:19: error: type alias redefinition with different
// types ('__half' vs 'half_float::half')." The reason is that the driver uses the definition of
// half from half.hpp, while rocrand uses definition of half type from HIP headers, i.e. the
// definitions are different.

#include <boost/core/demangle.hpp>
#include <half/half.hpp>

#include <iostream>
#include <typeinfo>

namespace gpumemrand {

int gen_0_1(double* buf, size_t sz);
int gen_0_1(float* buf, size_t sz);
int gen_0_1(half_float::half* buf, size_t sz);

template <typename T>
int gen_0_1(T* buf, size_t sz)
{
    std::cout << "Warning: gpumemrand functions are supported only for double, float and half. GPU "
                 "buffer { "
              << static_cast<void*>(buf) << ", " << sz << ", "
              << boost::core::demangle(typeid(T).name()) << " } remains uninitialized."
              << std::endl;
    return 0;
}

} // namespace gpumemrand

#endif // GUARD_ROCRAND_WRAPPER_HPP
