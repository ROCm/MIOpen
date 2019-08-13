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
#ifndef FLOAT_TYPES_HPP
#define FLOAT_TYPES_HPP

#include "bfloat16_dev.hpp"

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#ifdef __HIP_PLATFORM_HCC__
#define FLOAT half
#define FLOAT_ACCUM float
#else
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_ACCUM float
#endif                 // __HIP_PLATFORM_HCC__
#define SIZEOF_FLOAT 2 /* sizeof is unavailable for preprocessor */
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif // HALF_MAX
#endif // MIOPEN_USE_FP16

#if MIOPEN_USE_FP32 == 1
#ifdef __HIP_PLATFORM_HCC__
#define FLOAT float
#define FLOAT_ACCUM float
#else
#define _FLOAT float
#define _FLOAT_ACCUM float
#endif                 // __HIP_PLATFORM_HCC__
#define SIZEOF_FLOAT 4 /* sizeof is unavailable for preprocessor */
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif // FLT_MAX
#endif // MIOPEN_USE_FP32

#if MIOPEN_USE_BFP16 == 1
#ifdef __HIP_PLATFORM_HCC__
#define FLOAT ushort
#define FLOAT_ACCUM float
#else
#define _FLOAT ushort
#define _FLOAT_ACCUM float
#endif                 //
#define SIZEOF_FLOAT 2 /* sizeof is unavailable for preprocessor */
#define MAX_VAL 0x7F7F /* max value */
#endif                 // MIOPEN_USE_BFP16

#if MIOPEN_USE_FP16 == 1
#ifdef __HIP_PLATFORM_HCC__
#define CVT_FLOAT2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#define CVT_ACCUM2FLOAT(x) (static_cast<FLOAT>(x))
#else
#define CVT_FLOAT2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_ACCUM2FLOAT(x) ((_FLOAT)(x))
#endif // MIOPEN_BACKEND_HIP
#endif // MIOPEN_USE_FP16

#if MIOPEN_USE_FP32 == 1
#ifdef __HIP_PLATFORM_HCC__
#define CVT_FLOAT2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#define CVT_ACCUM2FLOAT(x) (static_cast<FLOAT>(x))
#else
#define CVT_FLOAT2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_ACCUM2FLOAT(x) ((_FLOAT)(x))
#endif
#endif // MIOPEN_USE_FP32

#if MIOPEN_USE_BFP16 == 1
#define CVT_FLOAT2ACCUM(x) bfloat16_to_float(x)
#define CVT_ACCUM2FLOAT(x) float_to_bfloat16(x)
#endif

#ifndef __HIP_PLATFORM_HCC__
#define _FLOAT2 PPCAT(_FLOAT, TWO)
#endif

#endif // FLOAT_TYPES_HPP
