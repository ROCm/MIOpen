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
#include "bfloat16.h"

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _FLOAT_ACCUM float
#define SIZEOF_FLOAT 2 /* sizeof is unavailable for preprocessor */
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define _FLOAT_ACCUM float
#define SIZEOF_FLOAT 4 /* sizeof is unavailable for preprocessor */
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif
#if MIOPEN_USE_BFP16 == 1
#define _FLOAT ushort
#define _FLOAT_ACCUM float
#define SIZEOF_FLOAT 2 /* sizeof is unavailable for preprocessor */
#define MAX_VAL 0x7F7F /* max value */
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#if MIOPEN_USE_FP16 == 1
#define CVT_FLOAT2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_ACCUM2FLOAT(x) ((_FLOAT)(x))
#endif
#if MIOPEN_USE_FP32 == 1
#define CVT_FLOAT2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_ACCUM2FLOAT(x) ((_FLOAT)(x))
#endif
#if MIOPEN_USE_BFP16 == 1
#define CVT_FLOAT2ACCUM(x) bfloat16_to_float(x)
#define CVT_ACCUM2FLOAT(x) float_to_bfloat16(x)
#endif
