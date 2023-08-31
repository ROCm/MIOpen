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
// #include <miopen/bfloat16.hpp>
// #include <half.hpp>
namespace miopen_hip_f8_impl {

#ifndef __HIP_PLATFORM_HCC__
using hip_bfloat16 = bfloat16;
using half         = half_float::half;
#endif

template <int wm, int we, typename T>
MIOPEN_HIP_HOST_DEVICE uint8_t cast_to_f8_no_range_reduce(T _x,
                                                          bool stoch   = false,
                                                          uint32_t rng = 0)
{
    static_assert(we == 5, "we==5");
    static_assert(sizeof(T) == 2, "no_range_reduce only works for float16");

    uint32_t x = *(reinterpret_cast<uint16_t*>(&_x));

    uint32_t head, mantissa, exponent;
    uint32_t sign;

    const int mfmt      = 10;
    head                = x & 0xFC00;
    mantissa            = x & 0x3FF;
    exponent            = (head >> 10) & 0x1F;
    sign                = head >> 15;
    uint32_t signed_inf = (sign << 7) + (((1 << we) - 1) << wm);

    if((x & 0x7FFF) == 0x7C00)
        return signed_inf;
    if((x & 0x7C00) == 0x7C00)
        return signed_inf + 1;
    if(x == 0)
        return 0;
    if(x == 0x8000)
        return 0x80;

    //  uint32_t nextbit = 1<<(mfmt-wm-1);
    uint32_t drop_mask = (1 << (mfmt - wm)) - 1;

    // const int max_exp = (1<<we)-(negative_zero_nan ? 1 : 2);
    mantissa += (stoch ? rng : mantissa) & drop_mask;
    if(exponent != 0)
        mantissa += 1 << mfmt;
    if(mantissa >= (2 << mfmt))
    {
        mantissa >>= 1;
        exponent++;
    }
    else if(mantissa >= (1 << mfmt) && exponent == 0)
    {
        exponent++;
    }
    mantissa >>= (mfmt - wm);
    mantissa &= (1 << wm) - 1;
    if(exponent == 31)
        return (sign << 7) | 0x7B;
    return (sign << 7) | (exponent << wm) | mantissa;
}

template <int wm, int we, typename T, bool negative_zero_nan, bool clip>
MIOPEN_HIP_HOST_DEVICE uint8_t cast_to_f8(T _x, bool stoch, uint32_t rng)
{
    constexpr bool is_half  = std::is_same<T, half>::value;
    constexpr bool is_float = std::is_same<T, float>::value;
    static_assert(wm + we == 7, "wm+we==7");
    static_assert(is_half || is_float, "Only half and float can be cast to f8");

    if(sizeof(T) == 2 && we == 5 && !negative_zero_nan)
        return cast_to_f8_no_range_reduce<2, 5, half>(static_cast<half>(_x), stoch, rng);

    const int mfmt = (sizeof(T) == 4) ? 23 : 10;
    uint32_t x;
    if(sizeof(T) == 4)
        x = *(reinterpret_cast<uint32_t*>(&_x)); // cppcheck-suppress invalidPointerCast
    else
        x = *(reinterpret_cast<uint16_t*>(&_x)); // cppcheck-suppress invalidPointerCast

    uint32_t head, mantissa;
    int exponent;
    uint32_t sign;

    if(sizeof(T) == 4)
    {
        head     = x & 0xFF800000;
        mantissa = x & 0x7FFFFF;
        exponent = (head >> 23) & 0xFF;
        sign     = head >> 31;
    }
    else
    {
        head     = x & 0xFC00;
        mantissa = x & 0x3FF;
        exponent = (head >> 10) & 0x1F;
        sign     = head >> 15;
    }

    uint32_t signed_inf = (sign << 7) + (((1 << we) - 1) << wm);

    if(negative_zero_nan)
    {
        if(sizeof(T) == 4)
        {
            if((x & 0x7F800000) == 0x7F800000)
                return 0x80;
        }
        else
        {
            // if(__hisinf(x) || __hisnan(x))
            if((x & 0x7C00) == 0x7C00)
                return 0x80;
        }
    }
    else
    {
        if(sizeof(T) == 4)
        {
            if((x & 0x7F800000) == 0x7F800000)
                return signed_inf + (mantissa != 0 ? 1 : 0);
        }
        else
        {
            if((x & 0x7C00) == 0x7C00)
                return signed_inf + (mantissa != 0 ? 1 : 0);
        }
    }
    if(x == 0)
        return 0;

    if(is_half && we == 5 && negative_zero_nan && exponent == 0)
    {
        exponent += 1;
        // TODO: call __clz when this is device code
        int sh = 1 + __builtin_clz(mantissa) - (32 - mfmt);
        mantissa <<= sh;
        exponent -= sh;
        /*
        while(mantissa < (1<<mfmt)) {
          mantissa <<= 1;
          exponent -= 1;
        }
        */
        mantissa &= ~(1 << mfmt);
    }

    uint32_t drop_mask = (1 << (mfmt - wm)) - 1;
    const int max_exp  = (1 << we) - (negative_zero_nan ? 1 : 2);
    const int exp_low_cutoff =
        (sizeof(T) == 4 ? 128 : 16) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

    exponent -= exp_low_cutoff - 1;
    if(exponent <= 0)
        drop_mask = (1 << (mfmt - wm + 1 - exponent)) - 1;
    mantissa += 1 << mfmt;
    mantissa += (stoch ? rng : mantissa) & drop_mask;
    if(mantissa >= (2 << mfmt))
    {
        mantissa >>= 1;
        exponent++;
    }
    mantissa >>= (mfmt - wm);

    if(exponent <= 0)
    {
        if(x == 0) // cppcheck-suppress identicalConditionAfterEarlyExit
            return 0;
        else
        {
            // subnormal range; represented by a subnormal float8 (exponent 0)
            // and involves loss of accuracy
            mantissa >>= 1 - exponent;
            exponent = 0;
        }
    }
    // above range: quantize to maximum possible float of the same sign
    else if(exponent > max_exp)
    {
        if(clip)
        {
            mantissa = (1 << wm) - 1;
            exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }
    if(exponent == 0 && mantissa == 0)
        return negative_zero_nan ? 0 : (sign << 7);
    mantissa &= (1 << wm) - 1;
    return (sign << 7) | (exponent << wm) | mantissa;
}

template <int wm, int we, typename T, bool negative_zero_nan>
MIOPEN_HIP_HOST_DEVICE T cast_from_f8(uint8_t x)
{
    constexpr bool is_half  = std::is_same<T, half>::value;
    constexpr bool is_float = std::is_same<T, float>::value;
    static_assert(is_half || is_float, "only half and float are supported");

    constexpr int weo = is_half ? 5 : 8;
    constexpr int wmo = is_half ? 10 : (is_float ? 23 : 7);

    T fInf, fNegInf, fNaN, fNeg0;
    if(is_half)
    {
        const uint16_t ihInf    = 0x7C00;
        const uint16_t ihNegInf = 0xFC00;
        const uint16_t ihNaN    = 0x7C01;
        const uint16_t ihNeg0   = 0x8000;
        fInf                    = *(reinterpret_cast<const half*>(&ihInf));
        fNegInf                 = *(reinterpret_cast<const half*>(&ihNegInf));
        fNaN                    = *(reinterpret_cast<const half*>(&ihNaN));
        fNeg0                   = *(reinterpret_cast<const half*>(&ihNeg0));
    }
    else if(is_float)
    {
        const uint32_t ifInf    = 0x7F800000;
        const uint32_t ifNegInf = 0xFF800000;
        const uint32_t ifNaN    = 0x7F800001;
        const uint32_t ifNeg0   = 0x80000000;
        fInf = *(reinterpret_cast<const float*>(&ifInf)); // cppcheck-suppress invalidPointerCast
        fNegInf =
            *(reinterpret_cast<const float*>(&ifNegInf));   // cppcheck-suppress invalidPointerCast
        fNaN  = *(reinterpret_cast<const float*>(&ifNaN));  // cppcheck-suppress invalidPointerCast
        fNeg0 = *(reinterpret_cast<const float*>(&ifNeg0)); // cppcheck-suppress invalidPointerCast
    }

    if(x == 0)
        return static_cast<T>(0);

    uint32_t sign     = x >> 7;
    uint32_t mantissa = x & ((1 << wm) - 1);
    int exponent      = (x & 0x7F) >> wm;
    if(negative_zero_nan)
    {
        if(x == 0x80)
            return fNaN;
    }
    else
    {
        if(x == 0x80)
            return fNeg0;
        if(exponent == ((1 << we) - 1))
            return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
    }
    typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type retval;
    if(we == 5 && is_half && !negative_zero_nan)
    {
        retval = x << 8;
        return *(reinterpret_cast<const T*>(&retval));
    }

    const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        int sh = 1 + __builtin_clz(mantissa) - (32 - wm);
        mantissa <<= sh;
        exponent += 1 - sh;
        /*
        exponent++;
        while(mantissa<(1<<wm)) {
          mantissa <<= 1;
          exponent--;
        }
        */
        mantissa &= ((1 << wm) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= wmo - wm;

    // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << wmo;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    if(sizeof(T) == 2)
        retval = (sign << 15) | (exponent << 10) | mantissa;
    else
        retval = (sign << 31) | (exponent << 23) | mantissa;
    return *(reinterpret_cast<const T*>(&retval));
}

} // namespace miopen_hip_f8_impl
