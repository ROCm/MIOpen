/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

/// This file will give all not defined macros default value

#ifndef DEFAULT_CONFIGURATIONS_HPP
#define DEFAULT_CONFIGURATIONS_HPP

// ---------- general configs ----------

// normalization of input macros (give default value to undefined macros)
// TODO: we can consider to remove all of these default values, force the
// user to define all of them.
#ifndef MIO_LAYOUT_NHWC
#define MIO_LAYOUT_NHWC 0 // Default value
#endif

#ifndef MIO_SAVE_MEAN_VARIANCE
#define MIO_SAVE_MEAN_VARIANCE 0
#endif

#ifndef MIO_RUNNING_RESULT
#define MIO_RUNNING_RESULT 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FPMIX
#define MIOPEN_USE_FPMIX 0
#endif

#ifndef MIOPEN_USE_BFPMIX
#define MIOPEN_USE_BFPMIX 0
#endif

#ifndef MIO_BN_NODPP
#define MIO_BN_NODPP 0
#endif

#ifndef MIOPEN_USE_AMDGCN
#define MIOPEN_USE_AMDGCN 1
#endif

#ifndef MIOPEN_NRN_OP_ID
#define MIOPEN_NRN_OP_ID 0
#endif

// ---------- batchnorm configs ----------

#ifndef HALF_MAX
#define HALF_MAX 65504
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

#ifndef MIO_BN_GRP0
#define MIO_BN_GRP0 1
#endif

#ifndef MIO_BN_GRP1
#define MIO_BN_GRP1 1
#endif

#ifndef MIO_BN_GRP2
#define MIO_BN_GRP2 1
#endif

#ifndef MIO_BN_GFX103X
#define MIO_BN_GFX103X 0
#endif

#ifndef MIO_BN_GFX110X
#define MIO_BN_GFX110X 0
#endif

#ifndef MIO_BN_GFX120X
#define MIO_BN_GFX120X 0
#endif

#ifndef MIO_BN_GFX115X
#define MIO_BN_GFX115X 0
#endif

#ifndef MIO_BN_VARIANT
#define MIO_BN_VARIANT 255
#endif

#ifndef MIO_BN_NCHW
#define MIO_BN_NCHW 1
#endif

#ifndef MIO_BN_MAXN
#define MIO_BN_MAXN 65
#endif

#ifndef MIO_BN_VECTORIZE
#define MIO_BN_VECTORIZE 0
#endif

#ifndef MIO_BN_STASH_METHOD
#define MIO_BN_STASH_METHOD 0
#endif

#ifndef MIO_BN_LOOP_UNROLL_MAXN
#define MIO_BN_LOOP_UNROLL_MAXN 768
#endif

#ifndef MIO_BN_LOOP_UNROLL_MAXHW
#define MIO_BN_LOOP_UNROLL_MAXHW 2500
#endif

#ifndef MIO_BN_LDSGCN_SIZE
#define MIO_BN_LDSGCN_SIZE 16
#endif

#ifndef MIO_BN_LDS_SIZE
#define MIO_BN_LDS_SIZE 256
#endif

#ifndef MIO_BN_NGRPS
#define MIO_BN_NGRPS 1
#endif

#ifndef MIO_BN_C
#define MIO_BN_C 1
#endif

#ifndef MIO_BN_N
#define MIO_BN_N 1
#endif

#ifndef MIO_BN_NHW
#define MIO_BN_NHW 1
#endif

#ifndef MIO_BN_INHW
#define MIO_BN_INHW 1
#endif

#ifndef MIO_BN_CHW
#define MIO_BN_CHW 1
#endif

#ifndef MIO_BN_HW
#define MIO_BN_HW 1
#endif

#endif
