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

#pragma once

namespace miopen {

//=============================================================================
// Float overloads
//=============================================================================

__forceinline__ __device__ float exp(float x) { return expf(x); }
__forceinline__ __device__ float log(float x) { return logf(x); }
__forceinline__ __device__ float sqrt(float x) { return sqrtf(x); }
__forceinline__ __device__ float rsqrt(float x) { return rsqrtf(x); }
__forceinline__ __device__ float sin(float x) { return sinf(x); }
__forceinline__ __device__ float cos(float x) { return cosf(x); }
__forceinline__ __device__ float tan(float x) { return tanf(x); }
__forceinline__ __device__ float tanh(float x) { return tanhf(x); }
__forceinline__ __device__ float pow(float x, float y) { return powf(x, y); }
__forceinline__ __device__ float fabs(float x) { return fabsf(x); }
__forceinline__ __device__ float fmax(float x, float y) { return fmaxf(x, y); }
__forceinline__ __device__ float fmin(float x, float y) { return fminf(x, y); }

//=============================================================================
// Half precision overloads
//=============================================================================

__forceinline__ __device__ _Float16 exp(_Float16 x) { return __ocml_exp_f16(x); }
__forceinline__ __device__ _Float16 log(_Float16 x) { return __ocml_log_f16(x); }
__forceinline__ __device__ _Float16 sqrt(_Float16 x) { return __ocml_sqrt_f16(x); }
__forceinline__ __device__ _Float16 rsqrt(_Float16 x) { return __ocml_rsqrt_f16(x); }
__forceinline__ __device__ _Float16 sin(_Float16 x) { return hsin(__half(x)); }
__forceinline__ __device__ _Float16 cos(_Float16 x) { return hcos(__half(x)); }
__forceinline__ __device__ _Float16 fabs(_Float16 x) { return __ocml_fabs_f16(x); }
__forceinline__ __device__ _Float16 fmin(_Float16 x, _Float16 y) { return __ocml_fmin_f16(x, y); }
__forceinline__ __device__ _Float16 fmax(_Float16 x, _Float16 y) { return __ocml_fmax_f16(x, y); }
__forceinline__ __device__ _Float16 pow(_Float16 x, _Float16 y)
{
    return __ocml_exp_f16(y * __ocml_log_f16(x));
}
__forceinline__ __device__ _Float16 tanh(_Float16 x)
{
    float x_scaled = static_cast<float>(x) * 1.4426950408889634f; // 0x1.715476p+0f = log2(e)
    float a        = __builtin_amdgcn_exp2f(x_scaled);
    float b        = __builtin_amdgcn_exp2f(-x_scaled);

    _Float16 ret = static_cast<_Float16>((a - b) * __builtin_amdgcn_rcpf(a + b));
    _Float16 one = __builtin_copysignf(1.0f, x);

    return __ocml_fabs_f16(x) > 4.5f ? one : ret;
}

} // namespace miopen
