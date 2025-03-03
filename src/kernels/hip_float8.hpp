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
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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

#include "miopen_cstdint.hpp"
#include "miopen_limits.hpp"

#ifndef MIOPEN_ENABLE_F8_DEVICE_CODE
#define MIOPEN_ENABLE_F8_DEVICE_CODE 0
#endif

// FP8 header version 0.4, 2021/05/11
// Updated by atamazov 2023/12/22
#if defined __HIP_PLATFORM_AMD__ && MIOPEN_ENABLE_F8_DEVICE_CODE
// MIOpen by default does not have device code in the regular compilation paths,
// therefore, when this file is used from the host side, compilation takes much
// longer. By guarding the __device__ directive we can control that such compilation
// only happens for kernels which include this file.
#define MIOPEN_HIP_HOST_DEVICE __host__ __device__
#else
#define MIOPEN_HIP_HOST_DEVICE
#endif

#define USE_SIMPLER_HIP_F8x8 0

#ifndef MIOPEN_FP8_CLIPPING
#define MIOPEN_FP8_CLIPPING 1
#endif

#ifndef MIOPEN_FP8_IEEE_EXPONENT_BIAS
#define MIOPEN_FP8_IEEE_EXPONENT_BIAS 1
#endif

namespace miopen_hip_f8_impl {

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
MIOPEN_HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch = false, uint32_t rng = 0);

template <int wm, int we, typename T, bool negative_zero_nan>
MIOPEN_HIP_HOST_DEVICE T cast_from_f8(uint8_t x);

} // namespace miopen_hip_f8_impl

#include "hip_f8_impl.hpp"

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

inline MIOPEN_HIP_HOST_DEVICE bool get_hip_f8_bias_mode()
{
#if MIOPEN_FP8_IEEE_EXPONENT_BIAS
    return false;
#else
    return true;
#endif
}

template <typename T>
class numeric_limits;

template <hip_f8_type T>
struct hip_f8
{
    uint8_t data;

    // default constructor
    MIOPEN_HIP_HOST_DEVICE hip_f8() = default;

    MIOPEN_HIP_HOST_DEVICE hip_f8(hip_f8<T> const&) = default;

    // constructor from bits
    explicit MIOPEN_HIP_HOST_DEVICE hip_f8(uint8_t v) { data = v; }

    // constructor from in
    explicit MIOPEN_HIP_HOST_DEVICE hip_f8(int v) : hip_f8(static_cast<float>(v)) {}

    explicit MIOPEN_HIP_HOST_DEVICE hip_f8(double v) : hip_f8(static_cast<float>(v)) {}

    // constructor from float
    explicit MIOPEN_HIP_HOST_DEVICE
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
    explicit MIOPEN_HIP_HOST_DEVICE
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
    explicit MIOPEN_HIP_HOST_DEVICE
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

    explicit MIOPEN_HIP_HOST_DEVICE hip_f8(hip_f8<T> v, hip_f8_rounding_mode, uint32_t)
    {
        this->data = v.data;
    }

    // constructor from hip_bfloat16
    explicit MIOPEN_HIP_HOST_DEVICE
    hip_f8(hip_bfloat16 v,
           hip_f8_rounding_mode r = miopen_f8::hip_f8_rounding_mode::standard,
           uint32_t rng           = 0);

    MIOPEN_HIP_HOST_DEVICE
    hip_f8& operator*=(const hip_f8& rhs)
    {
        const auto tmp = static_cast<float>(*this) * static_cast<float>(rhs);
        *this          = static_cast<hip_f8>(tmp);
        return *this;
    }

    MIOPEN_HIP_HOST_DEVICE
    hip_f8& operator+=(const hip_f8& rhs)
    {
        const auto tmp = static_cast<float>(*this) + static_cast<float>(rhs);
        *this          = static_cast<hip_f8>(tmp);
        return *this;
    }

    MIOPEN_HIP_HOST_DEVICE
    hip_f8& operator-=(const hip_f8& rhs)
    {
        const auto tmp = static_cast<float>(*this) - static_cast<float>(rhs);
        *this          = static_cast<hip_f8>(tmp);
        return *this;
    }

    inline MIOPEN_HIP_HOST_DEVICE hip_f8& operator=(const hip_f8& rhs)
    {
        if(&rhs != this)
            this->data = rhs.data;
        return *this;
    }

    inline MIOPEN_HIP_HOST_DEVICE bool operator==(const hip_f8& rhs) const
    {
        if((rhs.is_zero() && this->is_zero()) || (this->data == rhs.data))
        {
            return true;
        }
        else if(rhs.is_nan() || rhs.is_inf() || this->is_nan() || this->is_inf())
        {
            return false;
        }

        return false;
    }

    inline MIOPEN_HIP_HOST_DEVICE bool operator<(const hip_f8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we < them;
    }

    inline MIOPEN_HIP_HOST_DEVICE bool operator>(const hip_f8& rhs) const
    {
        const auto we   = static_cast<float>(*this);
        const auto them = static_cast<float>(rhs);
        return we > them;
    }
#if 0
    /*explicit*/ inline MIOPEN_HIP_HOST_DEVICE operator double()
    {
        // float tmp = static_cast<float>(*this);
        // return tmp;
    }

    /*explicit*/ inline MIOPEN_HIP_HOST_DEVICE operator double() const
    {
        // float tmp = static_cast<float>(*this);
        // return tmp;
    }
#endif
    // convert to float
    /*explicit*/ inline MIOPEN_HIP_HOST_DEVICE operator float() const
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
    /*explicit*/ inline MIOPEN_HIP_HOST_DEVICE operator half() const
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
    /*explicit*/ inline MIOPEN_HIP_HOST_DEVICE operator hip_bfloat16() const;

    // check for zero
    inline MIOPEN_HIP_HOST_DEVICE bool is_zero() const
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
    inline MIOPEN_HIP_HOST_DEVICE bool is_nan() const
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
    inline MIOPEN_HIP_HOST_DEVICE bool is_inf() const
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

template <miopen_f8::hip_f8_type T>
inline MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<T> operator*(miopen_f8::hip_f8<T> lhs,
                                                             const miopen_f8::hip_f8<T>& rhs)
{
    lhs *= rhs;
    return lhs;
}

template <miopen_f8::hip_f8_type T>
inline MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<T> operator+(miopen_f8::hip_f8<T> lhs,
                                                             const miopen_f8::hip_f8<T>& rhs)
{
    lhs += rhs;
    return lhs;
}

template <miopen_f8::hip_f8_type T>
inline MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<T> operator-(miopen_f8::hip_f8<T> lhs,
                                                             const miopen_f8::hip_f8<T>& rhs)
{
    lhs -= rhs;
    return lhs;
}

template <miopen_f8::hip_f8_type T, typename U>
inline MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<T> operator-(U lhs, const miopen_f8::hip_f8<T>& rhs)
{
    const auto tmp = static_cast<U>(rhs);
    return static_cast<miopen_f8::hip_f8<T>>(lhs - tmp);
}

template <miopen_f8::hip_f8_type T>
inline MIOPEN_HIP_HOST_DEVICE bool operator<(const miopen_f8::hip_f8<T>& lhs,
                                             const miopen_f8::hip_f8<T>& rhs)
{
    return static_cast<float>(lhs) < static_cast<float>(rhs);
}

template <miopen_f8::hip_f8_type T>
inline MIOPEN_HIP_HOST_DEVICE bool operator>(const miopen_f8::hip_f8<T>& lhs,
                                             const miopen_f8::hip_f8<T>& rhs)
{
    return static_cast<float>(lhs) > static_cast<float>(rhs);
}

template <miopen_f8::hip_f8_type T>
inline MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<T> fabs(miopen_f8::hip_f8<T> v)
{
#if !MIOPEN_FP8_IEEE_EXPONENT_BIAS
    // Preserve "universal" nan.
    // Otherwise it will be converted to valid 0.
    if(v.data == 0x80)
        return v;
#endif
    v.data = v.data & 0x7f;
    return v;
}

template <class T>
MIOPEN_HIP_HOST_DEVICE T Generate(uint8_t bits)
{
    union
    {
        uint8_t bits;
        T value;
    } x;

    x.bits = bits;
    return x.value;
}

template <>
class numeric_limits<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>
{
public:
    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> epsilon()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>(
            MIOPEN_FP8_IEEE_EXPONENT_BIAS ? 0x20 : 0x28); // 0.125
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> quiet_NaN()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>(
            MIOPEN_FP8_IEEE_EXPONENT_BIAS ? 0x7c : 0x80);
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> signaling_NaN()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>(
            MIOPEN_FP8_IEEE_EXPONENT_BIAS ? 0x79 : 0x80);
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> max()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>(
            MIOPEN_FP8_IEEE_EXPONENT_BIAS ? 0x77 : 0x7f); // 240
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> min()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>(0x08);
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> denorm_min()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>(0x01);
    }

    static constexpr int digits = 4;
};

template <>
class numeric_limits<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>
{
public:
    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> epsilon()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>(
            MIOPEN_FP8_IEEE_EXPONENT_BIAS ? 0x34 : 0x38); // 0.25
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> quiet_NaN()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>(
            MIOPEN_FP8_IEEE_EXPONENT_BIAS ? 0x7e : 0x80);
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> signaling_NaN()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>(
            MIOPEN_FP8_IEEE_EXPONENT_BIAS ? 0x7d : 0x80);
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> max()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>(
            MIOPEN_FP8_IEEE_EXPONENT_BIAS ? 0x7b : 0x7f); // 57344
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> min()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>(0x04);
    }

    static MIOPEN_HIP_HOST_DEVICE miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> denorm_min()
    {
        return miopen_f8::Generate<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>(0x01);
    }

    static constexpr int digits = 3;
};

} // namespace miopen_f8

#ifdef __HIPCC_RTC__
// Assume that if hipRTC is used, then we get <cmath> for F8
// from the precompiled header.
#else
// NOLINTBEGIN(cert-dcl58-cpp)
namespace std {
inline bool isfinite(miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> x) // NOLINT
{
    return !(x.is_inf() || x.is_nan());
}

inline bool isfinite(miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> x) // NOLINT
{
    return !(x.is_inf() || x.is_nan());
}

inline bool isnan(miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8> x) // NOLINT
{
    return x.is_nan();
}

inline bool isnan(miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8> x) // NOLINT
{
    return x.is_nan();
}

} // namespace std
  // NOLINTEND(cert-dcl58-cpp)
#endif

// NOLINTBEGIN(cert-dcl58-cpp)
namespace std {

template <>
class numeric_limits<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>
    : public miopen_f8::numeric_limits<miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>>
{
};

template <>
class numeric_limits<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>
    : public miopen_f8::numeric_limits<miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>>
{
};

} // namespace std
// NOLINTEND(cert-dcl58-cpp)

template <miopen_f8::hip_f8_type T>
struct hip_f8x4
{
    // define some convenience types
    using float32x2 = float __attribute__((ext_vector_type(2)));
    using float32x4 = float __attribute__((ext_vector_type(4)));

    using halfx2 = _Float16 __attribute__((ext_vector_type(2)));
    using halfx4 = _Float16 __attribute__((ext_vector_type(4)));

    using hip_bfloat16x2 = uint16_t __attribute__((ext_vector_type(2)));
    using hip_bfloat16x4 = uint16_t __attribute__((ext_vector_type(4)));

    uint32_t data;

    // default constructor
    MIOPEN_HIP_HOST_DEVICE hip_f8x4() = default;

    // constructor from bits
    MIOPEN_HIP_HOST_DEVICE hip_f8x4(uint32_t v);

    // constructor from float
    MIOPEN_HIP_HOST_DEVICE
    hip_f8x4(float v0,
             float v1                           = 0,
             float v2                           = 0,
             float v3                           = 0,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);
    MIOPEN_HIP_HOST_DEVICE
    hip_f8x4(float32x2 v,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);
    MIOPEN_HIP_HOST_DEVICE
    hip_f8x4(float32x4 v,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);

    // constructor from half
    MIOPEN_HIP_HOST_DEVICE
    hip_f8x4(half v0,
             half v1                            = {},
             half v2                            = {},
             half v3                            = {},
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);
    MIOPEN_HIP_HOST_DEVICE
    hip_f8x4(halfx2 v,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);
    MIOPEN_HIP_HOST_DEVICE
    hip_f8x4(halfx4 v,
             miopen_f8::hip_f8_rounding_mode rm = miopen_f8::hip_f8_rounding_mode::standard,
             uint32_t rng                       = 0);

    // convert to float32x4
    inline MIOPEN_HIP_HOST_DEVICE operator float32x4() const;

    // convert to halfx4
    inline MIOPEN_HIP_HOST_DEVICE operator halfx4() const;
};

template <miopen_f8::hip_f8_type T>
struct hip_f8x8
{
    // define some convenience types
    using f8x8 = hip_f8x4<T> __attribute__((ext_vector_type(2)));

    f8x8 data;

    // default constructor
    MIOPEN_HIP_HOST_DEVICE hip_f8x8() = default;

    // do we need to define other constructors or any conversion routines here?
};

// If we do not end up needing either any constructors or conversion routines for the above type,
// then we can simplify the above type to the following
#if USE_SIMPLER_HIP_F8x8
template <hip_f8_type T>
using hip_f8x8 = hip_f8x4<T> __attribute__((ext_vector_type(2)));
#endif

using hip_float32x4  = float __attribute__((ext_vector_type(4)));
using hip_float32x16 = float __attribute__((ext_vector_type(16)));

// these are device-specific and we don't expect them to exist unless we're compiling with hip-clang
// for MI300.
template <miopen_f8::hip_f8_type T_A, miopen_f8::hip_f8_type T_B>
__device__ hip_float32x4 mfma_f32_16x16x32(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x4 c);

template <miopen_f8::hip_f8_type T_A, miopen_f8::hip_f8_type T_B>
__device__ hip_float32x16 mfma_f32_32x32x16(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x16 c);

using float8_fnuz  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8_fnuz = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;
