/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef BFLOAT16_DEVICE_HPP
#define BFLOAT16_DEVICE_HPP

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __HIP_PLATFORM_AMD__
#define EXECUTION_SPECIFIER __device__
#else
#define EXECUTION_SPECIFIER
#endif // MIOPEN_BACKEND_HIP

typedef union cvt_bf16_fp32
{
    uint u32;
    ushort2 ushortx2;

// Composable kernels are written in HIP language. The language doesnt support
// ushort2.hi or ushort2.low.
#ifdef __HIP_PLATFORM_AMD__
    ushort ushortvec[2];
#endif // MIOPEN_BACKEND_HIP
    float f32;
} cvt_bf16_fp32_t;

EXECUTION_SPECIFIER float bfloat16_to_float(ushort src_val)
{
    cvt_bf16_fp32_t target_val;

#ifdef __HIP_PLATFORM_AMD__
    target_val.ushortx2 = make_ushort2(0, src_val);
#else
    target_val.ushortx2 = (ushort2)(0, src_val);
#endif

    return target_val.f32;
}

EXECUTION_SPECIFIER ushort float_to_bfloat16(float src_val)
{
    cvt_bf16_fp32_t target_val;
    target_val.f32 = src_val;
    // BF16 round and NaN preservation code matches
    // https://github.com/ROCm/rocBLAS/blob/develop/library/include/rocblas_bfloat16.h
    if((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        // When all of the exponent bits are 1, the value is Inf or NaN.
        // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
        // mantissa bit. Quiet NaN is indicated by the most significant mantissa
        // bit being 1. Signaling NaN is indicated by the most significant
        // mantissa bit being 0 but some other bit(s) being 1. If any of the
        // lower 16 bits of the mantissa are 1, we set the least significant bit
        // of the bfloat16 mantissa, in order to preserve signaling NaN in case
        // the bloat16's mantissa bits are all 0.
        if((target_val.u32 & 0xffff) != 0)
        {
            target_val.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    else
    {
#ifdef MIOPEN_USE_RNE_BFLOAT16
// When the exponent bits are not all 1s, then the value is zero, normal,
// or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
// 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
// This causes the bfloat16's mantissa to be incremented by 1 if the 16
// least significant bits of the float mantissa are greater than 0x8000,
// or if they are equal to 0x8000 and the least significant bit of the
// bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
// the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
// has the value 0x7f, then incrementing it causes it to become 0x00 and
// the exponent is incremented by one, which is the next higher FP value
// to the unrounded bfloat16 value. When the bfloat16 value is subnormal
// with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
// to a normal value with an exponent of 0x01 and a mantissa of 0x00.
// When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
// incrementing it causes it to become an exponent of 0xFF and a mantissa
// of 0x00, which is Inf, the next higher value to the unrounded value.
#ifdef __HIP_PLATFORM_AMD__
        target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
#else
        target_val.u32 +=
            (0x7fff + (target_val.ushortx2.hi & 1)); // Round to nearest, round to even
#endif // MIOPEN_BACKEND_HIP
#endif // MIOPEN_USE_RNE_BFLOAT16
    }

#ifdef __HIP_PLATFORM_AMD__
    return target_val.ushortvec[1];
#else
    return target_val.ushortx2.hi;
#endif // MIOPEN_BACKEND_HIP
}

#ifndef MIOPEN_USE_FP8
#define MIOPEN_USE_FP8 0
#endif

#ifndef MIOPEN_USE_BFP8
#define MIOPEN_USE_BFP8 0
#endif

#if MIOPEN_USE_FP8 || MIOPEN_USE_BFP8
// TODO: Convert the Col2Im kernels from OpenCL to HIP and remove the following
// functions which are rewrites of the f8 header impl functions
EXECUTION_SPECIFIER float fp8_to_float_impl(uchar x, const int wm, const int we)
{
    bool negative_zero_nan = MIOPEN_FP8_IEEE_EXPONENT_BIAS ? false : true;

    const int weo = 8;
    const int wmo = 23;

    float fInf, fNegInf, fNaN, fNeg0;
    const uint ifInf    = 0x7F800000;
    const uint ifNegInf = 0xFF800000;
    const uint ifNaN    = 0x7F800001;
    const uint ifNeg0   = 0x80000000;
    fInf                = *((const float*)(&ifInf));
    fNegInf             = *((const float*)(&ifNegInf));
    fNaN                = *((const float*)(&ifNaN));
    fNeg0               = *((const float*)(&ifNeg0));

    if(x == 0)
        return (float)(0);

    uint sign     = x >> 7;
    uint mantissa = x & ((1 << wm) - 1);
    int exponent  = (x & 0x7F) >> wm;
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
    uint retval;
    const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

    // subnormal input
    if(exponent == 0)
    {
        // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
        // TODO: verify __builtin_clz and OpenCL's clz do the same thing
        int sh = 1 + clz(mantissa) - (32 - wm);
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

    retval = (sign << 31) | (exponent << 23) | mantissa;
    return *((const float*)(&retval));
}

EXECUTION_SPECIFIER float fp8_to_float(uchar x) { return fp8_to_float_impl(x, 3, 4); }

EXECUTION_SPECIFIER float bfp8_to_float(uchar x) { return fp8_to_float_impl(x, 2, 5); }

inline uchar float_to_fp8_impl(float _x, const int wm, const int we) // bool stoch, uint rng)
{
    bool negative_zero_nan = MIOPEN_FP8_IEEE_EXPONENT_BIAS ? false : true;
    bool clip              = MIOPEN_FP8_CLIPPING;

    // Conserve the logic for stochastic rounding:
    bool stoch     = false;
    uint rng       = 0;
    const int mfmt = 23;
    uint x;
    x = *((uint*)(&_x));

    uint head, mantissa;
    int exponent;
    uint sign;

    head     = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head >> 23) & 0xFF;
    sign     = head >> 31;

    uint signed_inf = (sign << 7) + (((1 << we) - 1) << wm);

    if(negative_zero_nan)
    {
        if((x & 0x7F800000) == 0x7F800000)
            return 0x80;
    }
    else
    {
        if((x & 0x7F800000) == 0x7F800000)
            return signed_inf + (mantissa != 0 ? 1 : 0);
    }
    if(x == 0)
        return 0;

    uint drop_mask           = (1 << (mfmt - wm)) - 1;
    const int max_exp        = (1 << we) - (negative_zero_nan ? 1 : 2);
    const int exp_low_cutoff = (128) - (1 << (we - 1)) + 1 - (negative_zero_nan ? 1 : 0);

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

EXECUTION_SPECIFIER uchar float_to_fp8(float _x) // bool stoch, uint rng)
{
    return float_to_fp8_impl(_x, 3, 4);
}

EXECUTION_SPECIFIER uchar float_to_bfp8(float _x) // bool stoch, uint rng)
{
    return float_to_fp8_impl(_x, 2, 5);
}
#endif // MIOPEN_USE_FP8 || MIOPEN_USE_BFP8

#ifdef __cplusplus
}
#endif

#endif // BFLOAT16_DEVICE_HPP
