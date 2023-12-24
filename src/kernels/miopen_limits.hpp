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

#ifndef WORKAROUND_DO_NOT_USE_CUSTOM_LIMITS
#define WORKAROUND_DO_NOT_USE_CUSTOM_LIMITS 0
#endif

#if defined(MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS) && !WORKAROUND_DONT_USE_CUSTOM_LIMITS

#include <hip/hip_bfloat16.h>

#define MIOPEN_ENABLE_F8_DEVICE_CODE 1
#include "hip_float8.hpp"

namespace std {

template <typename T>
class numeric_limits;

template <>
class numeric_limits<float>
{
public:
    static constexpr __device__ float max() noexcept { return 0x1.FFFFFEp+127f; }

    static constexpr __device__ float min() noexcept { return 0x1p-126f; }
};

template <>
class numeric_limits<_Float16>
{
public:
    static constexpr __device__ _Float16 max() noexcept
    {
        return static_cast<_Float16>(0x1.FFCp+15f);
    }

    static constexpr __device__ _Float16 min() noexcept { return static_cast<_Float16>(0x1p-14f); }
};

template <>
class numeric_limits<hip_bfloat16>
{
public:
    static
#if HIP_PACKAGE_VERSION_FLAT >= 6000000000ULL
        constexpr
#endif
        __device__ hip_bfloat16
        max() noexcept
    {
        // data = 0x7F7F
        return static_cast<hip_bfloat16>(0x1.FEp+127f);
    }

    static
#if HIP_PACKAGE_VERSION_FLAT >= 6000000000ULL
        constexpr
#endif
        __device__ hip_bfloat16
        min() noexcept
    {
        // data = 0x0080
        return static_cast<hip_bfloat16>(0x1p-14f);
    }
};

} // namespace std

#else

#include <limits>

#endif
