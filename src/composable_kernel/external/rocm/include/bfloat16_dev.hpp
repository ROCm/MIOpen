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

#ifdef __HIP_PLATFORM_HCC__
#define EXECUTION_SPECIFIER __device__
#else
#define EXECUTION_SPECIFIER
#endif // MIOPEN_BACKEND_HIP

typedef union
{
    uint u32;
    ushort2 ushortx2;

// Composable kernels are written in HIP language. The language doesnt support
// ushort2.hi or ushort2.low.
#ifdef __HIP_PLATFORM_HCC__
    ushort ushortvec[2];
#endif // MIOPEN_BACKEND_HIP
    float f32;
} cvt_bf16_fp32_t;

EXECUTION_SPECIFIER float bfloat16_to_float(ushort src_val)
{
    cvt_bf16_fp32_t target_val;

#ifdef __HIP_PLATFORM_HCC__
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
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/include/rocblas_bfloat16.h
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
#ifdef __HIP_PLATFORM_HCC__
        target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
#else
        target_val.u32 +=
            (0x7fff + (target_val.ushortx2.hi & 1)); // Round to nearest, round to even
#endif // MIOPEN_BACKEND_HIP
#endif // MIOPEN_USE_RNE_BFLOAT16
    }

#ifdef __HIP_PLATFORM_HCC__
    return target_val.ushortvec[1];
#else
    return target_val.ushortx2.hi;
#endif // MIOPEN_BACKEND_HIP
}

#ifdef __cplusplus
}
#endif

#endif // BFLOAT16_DEVICE_HPP
