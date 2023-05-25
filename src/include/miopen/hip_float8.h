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
// FP8 header version 0.4, 2021/05/11
#ifdef __HIP_PLATFORM_HCC__
#define HIP_HOST_DEVICE __host__ __device__
#else
#define HIP_HOST_DEVICE
// #include <miopen/bfloat16.hpp>
#include <hip/hip_fp16.h>
#endif
#include <hip/hip_bfloat16.h>
#include <half.hpp>
//#ifndef __HIP_PLATFORM_HCC__
using half_float::half;
//#endif
#define USE_SIMPLER_HIP_F8x8 0

#ifndef MIOPEN_FP8_CLIPPING
#define MIOPEN_FP8_CLIPPING 1
#endif

#ifndef MIOPEN_FP8_IEEE_EXPONENT_BIAS
#define MIOPEN_FP8_IEEE_EXPONENT_BIAS 1
#endif

namespace miopen_hip_f8_impl {

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

template <int wm, int we, typename T, bool negative_zero_nan>
HIP_HOST_DEVICE T cast_from_f8(uint8_t x);

} // namespace miopen_hip_f8_impl

#include "hip_f8_impl.h"

namespace miopen_f8 {
enum class hip_f8_type
{
    bf8 = 0, // 1:5:2
    fp8 = 1  // 1:4:3
};

enum class hip_f8_rounding_mode
{
    standard,
    stochastic
};
} // namespace miopen_f8

#if 0
// bias mode bit implementation
//
// For MI100 simulation purpose, we keep a copy of it on the host and device
// (MI300 HW implementation will be different)
//
// The bias mode should only be accessed via its get/set routines.
// The set routine sets both copies to the same value, keeping them in sync
// The get routine will return the device copy for device functions and
// the host copy for host functions
//
// "bias mode optimial"
//    => "bias mode bit" = 1
//    => bias = 16 for 152, 8 for 143
//    => NAN/INF are represented as negative_zero
//
// "bias mode ieee"
//    => "bias mode bit" = 0
//    => bias = 15 for 152, 7 for 143
//    => NAN/INF are represented as per IEEE conventions

#ifdef __HIP_PLATFORM_HCC__
__device__ bool hip_f8_bias_mode_bit_device;
bool hip_f8_bias_mode_bit_host;
__global__ void set_hip_f8_bias_mode_bit(bool v) { hip_f8_bias_mode_bit_device = v; }

void set_hip_f8_bias_mode_ieee()
{
    hipLaunchKernelGGL(set_hip_f8_bias_mode_bit, dim3(1), dim3(1), 0, 0, false);
    hip_f8_bias_mode_bit_host = false;
}

void set_hip_f8_bias_mode_optimal()
{
    hipLaunchKernelGGL(set_hip_f8_bias_mode_bit, dim3(1), dim3(1), 0, 0, true);
    hip_f8_bias_mode_bit_host = true;
}
#endif
#endif

namespace miopen_f8 {
inline HIP_HOST_DEVICE bool get_hip_f8_bias_mode()
{
#if MIOPEN_FP8_IEEE_EXPONENT_BIAS
    return false;
#else
    return true;
#endif
}

template <hip_f8_type T>
struct hip_f8
{
    uint8_t data;

    // default constructor
    HIP_HOST_DEVICE hip_f8() = default;

    HIP_HOST_DEVICE hip_f8(hip_f8<T> const&) = default;

    // constructor from bits
    explicit HIP_HOST_DEVICE hip_f8(uint8_t v) { data = v; }

    // constructor from in
    explicit HIP_HOST_DEVICE hip_f8(int v) : hip_f8(static_cast<float>(v)) {}

    explicit HIP_HOST_DEVICE hip_f8(double v) : hip_f8(static_cast<float>(v)) {}

    // constructor from float
    explicit HIP_HOST_DEVICE
    hip_f8(float v,
           miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
           uint32_t rng                       = 0)
    {
        if(T == hip_f8_type::bf8)
        {
            if(get_hip_f8_bias_mode())
            {
                data = miopen_hip_f8_impl::cast_to_f8<2,
                                                      5,
                                                      float,
                                                      true /*negative_zero_nan*/,
                                                      MIOPEN_FP8_CLIPPING /*clip*/>(
                    v, (rm == miopen_f8::hip_f8_rounding_mode::stochastic), rng);
            }
            else
            {
                data = miopen_hip_f8_impl::cast_to_f8<2,
                                                      5,
                                                      float,
                                                      false /*negative_zero_nan*/,
                                                      MIOPEN_FP8_CLIPPING /*clip*/>(
                    v, (rm == miopen_f8::hip_f8_rounding_mode::stochastic), rng);
            }
        }
        else /* fp8*/
        {
            if(get_hip_f8_bias_mode())
            {
                data = miopen_hip_f8_impl::cast_to_f8<3,
                                                      4,
                                                      float,
                                                      true /*negative_zero_nan*/,
                                                      MIOPEN_FP8_CLIPPING /*clip*/>(
                    v, (rm == miopen_f8::hip_f8_rounding_mode::stochastic), rng);
            }
            else
            {
                data = miopen_hip_f8_impl::cast_to_f8<3,
                                                      4,
                                                      float,
                                                      false /*negative_zero_nan*/,
                                                      MIOPEN_FP8_CLIPPING /*clip*/>(
                    v, (rm == miopen_f8::hip_f8_rounding_mode::stochastic), rng);
            }
        }
    }

    // constructor from half
    explicit HIP_HOST_DEVICE
    hip_f8(half v,
           miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
           uint32_t rng                       = 0)
    {
        if(T == hip_f8_type::bf8)
        {
            if(get_hip_f8_bias_mode())
            {
                data = miopen_hip_f8_impl::cast_to_f8<2,
                                                      5,
                                                      half,
                                                      true /*negative_zero_nan*/,
                                                      MIOPEN_FP8_CLIPPING /*clip*/>(
                    v, (rm == miopen_f8::hip_f8_rounding_mode::stochastic), rng);
            }
            else
            {
                data = miopen_hip_f8_impl::cast_to_f8<2,
                                                      5,
                                                      half,
                                                      false /*negative_zero_nan*/,
                                                      MIOPEN_FP8_CLIPPING /*clip*/>(
                    v, (rm == miopen_f8::hip_f8_rounding_mode::stochastic), rng);
            }
        }
        else /* fp8*/
        {
            if(get_hip_f8_bias_mode())
            {
                data = miopen_hip_f8_impl::cast_to_f8<3,
                                                      4,
                                                      half,
                                                      true /*negative_zero_nan*/,
                                                      MIOPEN_FP8_CLIPPING /*clip*/>(
                    v, (rm == miopen_f8::hip_f8_rounding_mode::stochastic), rng);
            }
            else
            {
                data = miopen_hip_f8_impl::cast_to_f8<3,
                                                      4,
                                                      half,
                                                      false /*negative_zero_nan*/,
                                                      MIOPEN_FP8_CLIPPING /*clip*/>(
                    v, (rm == miopen_f8::hip_f8_rounding_mode::stochastic), rng);
            }
        }
    }
    template <hip_f8_type U>
    explicit HIP_HOST_DEVICE
    hip_f8(hip_f8<U> v,
           miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
           uint32_t rng                       = 0)
    {
        if(T == U)
        {
            data = v.data;
        }
        else
        {
            const auto tmp  = static_cast<float>(v);
            const auto tmp2 = hip_f8<U>(tmp, rm, rng);
            data            = tmp2.data;
        }
    }

    explicit HIP_HOST_DEVICE hip_f8(hip_f8<T> v, hip_f8_rounding_mode, uint32_t)
    {
        this->data = v.data;
    }

    // constructor from hip_bfloat16
    explicit HIP_HOST_DEVICE
    hip_f8(hip_bfloat16 v,
           hip_f8_rounding_mode r = miopen_f8::hip_f8_rounding_mode::standard,
           uint32_t rng           = 0);

    hip_f8& operator*=(const hip_f8& rhs)
    {
        const auto tmp = static_cast<float>(*this) * static_cast<float>(rhs);
        *this          = static_cast<hip_f8>(tmp);
        return *this;
    }

    hip_f8& operator+=(const hip_f8& rhs)
    {
        const auto tmp = static_cast<float>(*this) + static_cast<float>(rhs);
        *this          = static_cast<hip_f8>(tmp);
        return *this;
    }

    hip_f8& operator-=(const hip_f8& rhs)
    {
        const auto tmp = static_cast<float>(*this) - static_cast<float>(rhs);
        *this          = static_cast<hip_f8>(tmp);
        return *this;
    }

    void operator=(const hip_f8& rhs) { this->data = rhs.data; }

    bool operator==(const hip_f8& rhs) const
    {
        if(rhs.is_zero() && this->is_zero())
            return true;
        else if(rhs.is_nan() || rhs.is_inf() || this->is_nan() || this->is_inf())
            return false;
        else if(fabs(rhs - *this) < std::numeric_limits<hip_f8<T>>::epsilon())
            return true;
        else
            return false;
    }

    inline HIP_HOST_DEVICE bool operator<(const hip_f8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we < them;
    }

    inline HIP_HOST_DEVICE bool operator>(const hip_f8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we > them;
    }

    explicit inline HIP_HOST_DEVICE operator double()
    {
        return static_cast<double>(static_cast<float>(*this));
    }

    explicit inline HIP_HOST_DEVICE operator double() const
    {
        return static_cast<double>(static_cast<float>(*this));
    }

    // convert to float
    explicit inline HIP_HOST_DEVICE operator float() const
    {
        if(T == hip_f8_type::bf8)
        {
            if(get_hip_f8_bias_mode())
            {
                return miopen_hip_f8_impl::cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(
                    data);
            }
            else
            {
                return miopen_hip_f8_impl::cast_from_f8<2, 5, float, false /*negative_zero_nan*/>(
                    data);
            }
        }
        else /* fp8*/
        {
            if(get_hip_f8_bias_mode())
            {
                return miopen_hip_f8_impl::cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(
                    data);
            }
            else
            {
                return miopen_hip_f8_impl::cast_from_f8<3, 4, float, false /*negative_zero_nan*/>(
                    data);
            }
        }
    }

    // convert to half
    explicit inline HIP_HOST_DEVICE operator half() const
    {
        if(T == hip_f8_type::bf8)
        {
            if(get_hip_f8_bias_mode())
            {
                return miopen_hip_f8_impl::cast_from_f8<2, 5, half, true /*negative_zero_nan*/>(
                    data);
            }
            else
            {
                return miopen_hip_f8_impl::cast_from_f8<2, 5, half, false /*negative_zero_nan*/>(
                    data);
            }
        }
        else /* fp8*/
        {
            if(get_hip_f8_bias_mode())
            {
                return miopen_hip_f8_impl::cast_from_f8<3, 4, half, true /*negative_zero_nan*/>(
                    data);
            }
            else
            {
                return miopen_hip_f8_impl::cast_from_f8<3, 4, half, false /*negative_zero_nan*/>(
                    data);
            }
        }
    }

    // convert to hip_bfloat16
    explicit inline HIP_HOST_DEVICE operator hip_bfloat16() const;

    // check for zero
    inline HIP_HOST_DEVICE bool is_zero() const
    {
        if(get_hip_f8_bias_mode())
        {
            return data == 0x00;
        }
        else
        {
            return (data == 0x00) || (data == 0x80);
        }
    }

    // check for nan
    inline HIP_HOST_DEVICE bool is_nan() const
    {
        if(get_hip_f8_bias_mode())
        {
            return data == 0x80;
        }
        else
        {
            if(T == hip_f8_type::bf8)
            {
                return (data == 0x7d) || (data == 0x7e) || (data == 0x7f) || (data == 0xfd) ||
                       (data == 0xfe) || (data == 0xff);
            }
            else
            {
                return (data == 0x79) || (data == 0x7a) || (data == 0x7b) || (data == 0x7c) ||
                       (data == 0x7d) || (data == 0x7e) || (data == 0x7f) || (data == 0xf9) ||
                       (data == 0xfa) || (data == 0xfb) || (data == 0xfc) || (data == 0xfd) ||
                       (data == 0xfe) || (data == 0xff);
            }
        }
    }

    // check for inf
    inline HIP_HOST_DEVICE bool is_inf() const
    {
        if(get_hip_f8_bias_mode())
        {
            return data == 0x80;
        }
        else
        {
            if(T == hip_f8_type::bf8)
            {
                return (data == 0x7c) || (data == 0xfc);
            }
            else
            {
                return (data == 0x78) || (data == 0xf8);
            }
        }
    }
}; // end of class hip_f8
} // namespace miopen_f8

// define numeric limits for the new data type
namespace std {
inline bool isfinite(miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> x) { return x.is_inf(); }

inline bool isfinite(miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> x) { return x.is_inf(); }

template <class T>
T F8_Max(void)
{
    union
    {
        uint8_t bits;
        T value;
    } x;

    x.bits = 0x7F;
    return x.value;
}

template <>
class numeric_limits<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>
{
public:
    static miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> epsilon()
    {
        return static_cast<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>(float(0.0625));
    }

    static miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> quiet_NaN()
    {
        return static_cast<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>(
            static_cast<uint8_t>(miopen_f8::get_hip_f8_bias_mode() ? 0X80 : 0x79));
    }

    static miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> max()
    {
        return static_cast<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>(
            F8_Max<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>());
    }
};

template <>
class numeric_limits<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>
{
public:
    static miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> epsilon()
    {
        return static_cast<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>(float(0.125));
    }

    static miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> quiet_NaN()
    {
        return static_cast<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>(
            static_cast<uint8_t>(miopen_f8::get_hip_f8_bias_mode() ? 0X80 : 0x7d));
    }

    static miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> max()
    {
        return static_cast<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>(
            F8_Max<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>());
    }
};
} // namespace std

template <miopen_f8::hip_f8_type T>
inline HIP_HOST_DEVICE miopen_f8::hip_f8<T> operator*(miopen_f8::hip_f8<T> lhs,
                                                      const miopen_f8::hip_f8<T>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <miopen_f8::hip_f8_type T>
inline HIP_HOST_DEVICE miopen_f8::hip_f8<T> operator+(miopen_f8::hip_f8<T> lhs,
                                                      const miopen_f8::hip_f8<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <miopen_f8::hip_f8_type T>
inline HIP_HOST_DEVICE miopen_f8::hip_f8<T> operator-(miopen_f8::hip_f8<T> lhs,
                                                      const miopen_f8::hip_f8<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <miopen_f8::hip_f8_type T, typename U>
inline HIP_HOST_DEVICE miopen_f8::hip_f8<T> operator-(U lhs, const miopen_f8::hip_f8<T>& rhs)
{
    const auto tmp = static_cast<U>(rhs);
    return static_cast<miopen_f8::hip_f8<T>>(lhs - tmp);
}

template <miopen_f8::hip_f8_type T>
inline HIP_HOST_DEVICE bool operator<(const miopen_f8::hip_f8<T>& lhs,
                                      const miopen_f8::hip_f8<T>& rhs)
{
    return static_cast<float>(lhs) < static_cast<float>(rhs);
}

template <miopen_f8::hip_f8_type T>
inline HIP_HOST_DEVICE bool operator>(const miopen_f8::hip_f8<T>& lhs,
                                      const miopen_f8::hip_f8<T>& rhs)
{
    return static_cast<float>(lhs) > static_cast<float>(rhs);
}

namespace std {
template <miopen_f8::hip_f8_type T>
inline HIP_HOST_DEVICE miopen_f8::hip_f8<T> fabs(miopen_f8::hip_f8<T> v)
{
    v.data = v.data & 0x7f;
    return v;
}
} // namespace std

template <miopen_f8::hip_f8_type T>
struct hip_f8x4
{
    // define some convenience types
    typedef float float32x2 __attribute__((ext_vector_type(2)));
    typedef float float32x4 __attribute__((ext_vector_type(4)));

    typedef _Float16 halfx2 __attribute__((ext_vector_type(2)));
    typedef _Float16 halfx4 __attribute__((ext_vector_type(4)));

    typedef uint16_t hip_bfloat16x2 __attribute__((ext_vector_type(2)));
    typedef uint16_t hip_bfloat16x4 __attribute__((ext_vector_type(4)));

    uint32_t data;

    // default constructor
    HIP_HOST_DEVICE hip_f8x4() = default;

    // constructor from bits
    HIP_HOST_DEVICE hip_f8x4(uint32_t v);

    // constructor from float
    HIP_HOST_DEVICE
    hip_f8x4(float v0,
             float v1                           = 0,
             float v2                           = 0,
             float v3                           = 0,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);
    HIP_HOST_DEVICE
    hip_f8x4(float32x2 v,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);
    HIP_HOST_DEVICE
    hip_f8x4(float32x4 v,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);

    // constructor from half
    HIP_HOST_DEVICE
    hip_f8x4(half v0,
             half v1                            = {},
             half v2                            = {},
             half v3                            = {},
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);
    HIP_HOST_DEVICE
    hip_f8x4(halfx2 v,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);
    HIP_HOST_DEVICE
    hip_f8x4(halfx4 v,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);

    // constructor from hip_bfloat16
#if 0
    HIP_HOST_DEVICE hip_f8x4(hip_bfloat16 v0,
                             hip_bfloat16 v1         = hip_bfloat16(0.0f),
                             hip_bfloat16 v2         = hip_bfloat16(0.0f),
                             hip_bfloat16 v3         = hip_bfloat16(0.0f),
                             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
                             uint32_t rng            = 0);
    HIP_HOST_DEVICE hip_f8x4(hip_bfloat16x2 v,
                             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
                             uint32_t rng            = 0);
    HIP_HOST_DEVICE hip_f8x4(hip_bfloat16x4 v,
                             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
                             uint32_t rng            = 0);
#endif

    // convert to float32x4
    inline HIP_HOST_DEVICE operator float32x4() const;

    // convert to halfx4
    inline HIP_HOST_DEVICE operator halfx4() const;
#if 0
    // convert to hip_bfloat16x4
    inline HIP_HOST_DEVICE operator hip_bfloat16x4() const;
#endif
};

template <miopen_f8::hip_f8_type T>
struct hip_f8x8
{
    // define some convenience types
    typedef hip_f8x4<T> f8x8 __attribute__((ext_vector_type(2)));

    f8x8 data;

    // default constructor
    HIP_HOST_DEVICE hip_f8x8() = default;

    // do we need to define other constructors or any conversion routines here?
};

// If we do not end up needing either any constructors or conversion routines for the above type,
// then we can simplify the above type to the following
#if USE_SIMPLER_HIP_F8x8
template <hip_f8_type T>
using hip_f8x8 = hip_f8x4<T> __attribute__((ext_vector_type(2)));
#endif

typedef float hip_float32x4 __attribute__((ext_vector_type(4)));
typedef float hip_float32x16 __attribute__((ext_vector_type(16)));

// these are device-specific and we don't expect them to exist unless we're compiling with hip-clang
// for MI300.
template <miopen_f8::hip_f8_type T_A, miopen_f8::hip_f8_type T_B>
__device__ hip_float32x4 mfma_f32_16x16x32(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x4 c);

template <miopen_f8::hip_f8_type T_A, miopen_f8::hip_f8_type T_B>
__device__ hip_float32x16 mfma_f32_32x32x16(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x16 c);

using float8  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8 = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;
