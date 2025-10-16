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
#ifndef GUARD_POOLING_FUNCTIONS_H
#define GUARD_POOLING_FUNCTIONS_H

#define UNUSED __attribute__((__unused__))

#ifndef MLO_POOLING_INDEX_TYPE
#error "MLO_POOLING_INDEX_TYPE not defined"
#else
typedef MLO_POOLING_INDEX_TYPE index_t;
#endif

#define MLO_POOLING_OP_AVE 0
#define MLO_POOLING_OP_MAX 1
#define MLO_POOLING_OP_STC 2
#define MLO_POOLING_OP_AVE_INCLUSIVE 3

#ifndef MLO_POOLING_OP_ID
#define MLO_POOLING_OP_ID 0
#endif

__device__ constexpr _Float16 poolingMax(_Float16 a, _Float16 b) { return __builtin_fmaxf16(a, b); }

__device__ constexpr float poolingMax(float a, float b) { return fmaxf(a, b); }

__device__ float approxRcp(float x)
{
    // By default, the compiler is convervative about emitting v_rcp_f32.
    // This is because:
    // 1. The inputs are required to be normalized. This should be the
    //    case for most float operations that result from other float
    //    operations.
    // 2. The accuracy is 1 ULP. This is fine for OpenCL, where the
    //    required accuracy is only 2.5 ULP, but not for HIP.
    // The performance difference between v_rcp_f32 and actual division
    // is quite significant, hence this function for cases where 1 ULP
    // is close enough.
    return __builtin_amdgcn_rcpf(x);
}

#endif // GUARD_POOLING_FUNCTIONS_H
