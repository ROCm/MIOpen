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
#ifndef GUARD_FLOAT_TYPES_H
#define GUARD_FLOAT_TYPES_H

#include "bfloat16_dev.hpp"

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8
#if MIOPEN_USE_FP8 == 1
#ifdef __HIP_PLATFORM_AMD__
#define FLOAT hip_f8<miopen_f8::hip_f8_type::fp8>
#define FLOAT_ACCUM float
// HIP implements the correct operators for conversion

#else
#define _FLOAT uchar
#define _FLOAT_ACCUM float
// OpenCL requires explicit functions
#define CVT_FLOAT2ACCUM(x) fp8_to_float(x)
#define CVT_ACCUM2FLOAT(x) float_to_fp8(x)
#endif
#define SIZEOF_FLOAT 1
// Max value for the main datatype
#define MAX_VAL 0x7F
// Max value for accumulator
// #ifndef FLT_MAX
// #define MAX_VAL_ACCUM 3.402823466e+38F
// #else
// #define MAX_VAL_ACCUM FLT_MAX
// #endif
#endif // MIOPEN_USE_FP8

#if MIOPEN_USE_BFP8 == 1
#ifdef __HIP_PLATFORM_AMD__
#define FLOAT hip_f8<miopen_f8::hip_f8_type::bf8>
#define FLOAT_ACCUM float
#else
#define _FLOAT uchar
#define _FLOAT_ACCUM float
// OpenCL requires explicit functions
#define CVT_FLOAT2ACCUM(x) bfp8_to_float(x)
#define CVT_ACCUM2FLOAT(x) float_to_bfp8(x)
#endif
#define SIZEOF_FLOAT 1
// Max value for the main datatype
#define MAX_VAL 0x7F
// Max value for accumulator
// #ifndef FLT_MAX
// #define MAX_VAL_ACCUM 3.402823466e+38F
// #else
// #define MAX_VAL_ACCUM FLT_MAX
// #endif
#endif // MIOPEN_USE_BFP8

#ifndef __HIP_PLATFORM_AMD__
#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)
#endif

/// If MIOPEN_USE_DOUBLE_ACCUM is defined as 1 when "float_types.h" is included,
/// then all the ACCUM macros (the represent operations and types) will use FP64
/// instead of FP32. In other words, the computations will be
/// performed using the native datatype even if ACCUM macros are used.
/// This functionality is indended mostly for debugging.
#ifdef MIOPEN_USE_DOUBLE_ACCUM
#if !(MIOPEN_USE_DOUBLE_ACCUM == 0 || MIOPEN_USE_DOUBLE_ACCUM == 1)
#error "Invalid value of MIOPEN_USE_DOUBLE_ACCUM"
#endif
#else
#define MIOPEN_USE_DOUBLE_ACCUM 0
#endif

#if MIOPEN_USE_DOUBLE_ACCUM
#ifdef __HIP_PLATFORM_AMD__
#define FLOAT_ACCUM double
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define _FLOAT_ACCUM double
#endif // __HIP_PLATFORM_AMD__
#define MAX_VAL_ACCUM DBL_MAX
#else // MIOPEN_USE_DOUBLE_ACCUM
#ifdef __HIP_PLATFORM_AMD__
#define FLOAT_ACCUM float
#else
#define _FLOAT_ACCUM float
#endif // __HIP_PLATFORM_AMD__
#ifndef FLT_MAX
#define MAX_VAL_ACCUM 3.402823466e+38F
#else
#define MAX_VAL_ACCUM FLT_MAX
#endif
#endif // MIOPEN_USE_DOUBLE_ACCUM

#if MIOPEN_USE_FP16 == 1
#ifdef __HIP_PLATFORM_AMD__
#define FLOAT _Float16
#else // __HIP_PLATFORM_AMD__
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#endif // __HIP_PLATFORM_AMD__
#define SIZEOF_FLOAT 2
// Max value for the main datatype
#ifndef HALF_MAX
#define MAX_VAL 65504
#else
#define MAX_VAL HALF_MAX
#endif
#endif // MIOPEN_USE_FP16

#if MIOPEN_USE_FP32 == 1
#ifdef __HIP_PLATFORM_AMD__
#define FLOAT float
#else
#define _FLOAT float
#endif // __HIP_PLATFORM_AMD__
#define SIZEOF_FLOAT 4
// Max value for the main datatype
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F
#else
#define MAX_VAL FLT_MAX
#endif
#endif // MIOPEN_USE_FP32

#if MIOPEN_USE_BFP16 == 1
#ifdef __HIP_PLATFORM_AMD__
#define FLOAT ushort
#else
#define _FLOAT ushort
#endif //
#define SIZEOF_FLOAT 2
// Max value for the main datatype
#define MAX_VAL 0x7F7F
#endif // MIOPEN_USE_BFP16

#if MIOPEN_USE_FP16 == 1
#ifdef __HIP_PLATFORM_AMD__
#define CVT_FLOAT2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#define CVT_ACCUM2FLOAT(x) (static_cast<FLOAT>(x))
#define CVT_INTEGRAL2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#else
#define CVT_FLOAT2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_ACCUM2FLOAT(x) ((_FLOAT)(x))
#define CVT_INTEGRAL2ACCUM(x) ((_FLOAT_ACCUM)(x))
#endif
// These two are required to uniformly initialize
// variables with non-zero literal constants of FP32 type
// regardless of the actual type of the variable.
// This is especially complicated for BF16, because
// the compiler lacks the support of BF16 literals.
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#endif // MIOPEN_USE_FP16

#if MIOPEN_USE_FP32 == 1
/// \todo Basically, conversions from float to accum and vice versa
/// should be removed because FLOAT_ACCUM and FLOAT are identical.
/// However this may lead to problems if these macros are used in
/// inappropriate contexts (e.g. with integral types), so this
/// refactoring should be considered as nontrivial and requires
/// a separate PR. Let's keep this historical stuff for now.
/// --atamazov 30.08.2023
#ifdef __HIP_PLATFORM_AMD__
#define CVT_FLOAT2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#define CVT_ACCUM2FLOAT(x) (static_cast<FLOAT>(x))
#define CVT_INTEGRAL2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#else
#define CVT_FLOAT2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_ACCUM2FLOAT(x) ((_FLOAT)(x))
#define CVT_INTEGRAL2ACCUM(x) ((_FLOAT_ACCUM)(x))
#endif
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#endif // MIOPEN_USE_FP32

#if MIOPEN_USE_BFP16 == 1
#ifdef __HIP_PLATFORM_AMD__
#define CVT_FLOAT2ACCUM(x) (bfloat16_to_float(x))
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#define CVT_INTEGRAL2ACCUM(x) (static_cast<FLOAT_ACCUM>(x))
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#else
#define CVT_FLOAT2ACCUM(x) (bfloat16_to_float(x))
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#define CVT_INTEGRAL2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#endif
#endif

/// If MIOPEN_USE_NATIVE_DATATYPE_ACCUM is defined as 1 when "float_types.h" is included,
/// then all the ACCUM macros (the represent operations and types) will use the native
/// datatype (BF16 or FP16) instead of FP32. In other words, the computations will be
/// performed using the native datatype even if ACCUM macros are used. This allows for
/// building both mixed-precision and "pure" kernels from the single source.
/// Note: This macro has higher priority than MIOPEN_USE_DOUBLE_ACCUM.
#ifdef MIOPEN_USE_NATIVE_DATATYPE_ACCUM
#if !(MIOPEN_USE_NATIVE_DATATYPE_ACCUM == 0 || MIOPEN_USE_NATIVE_DATATYPE_ACCUM == 1)
#error "Invalid value of MIOPEN_USE_NATIVE_DATATYPE_ACCUM"
#endif
#else
#define MIOPEN_USE_NATIVE_DATATYPE_ACCUM 0
#endif

#if MIOPEN_USE_NATIVE_DATATYPE_ACCUM
#ifdef __HIP_PLATFORM_AMD__
#undef FLOAT_ACCUM
#define FLOAT_ACCUM MIOPEN_ERROR_NOT_IMLEMENTED
#else
#undef _FLOAT_ACCUM
#define _FLOAT_ACCUM _FLOAT
#endif

#undef MAX_VAL_ACCUM
#define MAX_VAL_ACCUM MAX_VAL
#undef CVT_FLOAT2ACCUM
#define CVT_FLOAT2ACCUM(x) (x)
#undef CVT_ACCUM2FLOAT
#define CVT_ACCUM2FLOAT(x) (x)
#undef CVT_FP32_2ACCUM
#define CVT_FP32_2ACCUM(x) (CVT_FP32_2FLOAT(x))

#undef CVT_INTEGRAL2ACCUM
#ifdef __HIP_PLATFORM_AMD__
#define CVT_INTEGRAL2ACCUM(x) MIOPEN_ERROR_NOT_IMLEMENTED
#else
#if MIOPEN_USE_BFP16 == 1
// No direct conversion from integral types to BF16 is available.
// WARNING: Precision loss when integral type is wider than 16 bits.
#define CVT_INTEGRAL2ACCUM(x) (float_to_bfloat16(static_cast<float>(x)))
#else
#define CVT_INTEGRAL2ACCUM(x) ((_FLOAT)(x))
#endif
#endif
#endif // MIOPEN_USE_NATIVE_DATATYPE_ACCUM

#endif // GUARD_FLOAT_TYPES_H
