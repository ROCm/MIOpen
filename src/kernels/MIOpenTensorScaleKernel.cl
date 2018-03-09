/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)
#define _AS_FLOAT PPCAT(as_, _FLOAT)

#ifndef GLOBAL_WORK_SIZE_X
#define GLOBAL_WORK_SIZE_X 1
#endif

#ifndef GLOBAL_WORK_SIZE_Y
#define GLOBAL_WORK_SIZE_Y 1
#endif

#ifndef GLOBAL_WORK_SIZE_Z
#define GLOBAL_WORK_SIZE_Z 1
#endif

__kernel void ScaleTensor1d(global _FLOAT* __restrict dst,
                            const _FLOAT alpha,
                            const int offset,
                            const int stride0,
                            const int len0)
{
    const uint tidx = get_global_id(0);

    for(uint did0 = tidx; did0 < len0; did0 += GLOBAL_WORK_SIZE_X)
    {
        const uint i = stride0 * did0;

        dst[i + offset] *= alpha;
    }
}

__kernel void ScaleTensor2d(global _FLOAT* __restrict dst,
                            const _FLOAT alpha,
                            const int offset,
                            const int stride0,
                            const int stride1,
                            const int len0,
                            const int len1)
{
    const uint tidx = get_global_id(0);
    const uint tidy = get_global_id(1);

    for(uint did0 = tidy; did0 < len0; did0 += GLOBAL_WORK_SIZE_Y)
    {
        for(uint did1 = tidx; did1 < len1; did1 += GLOBAL_WORK_SIZE_X)
        {
            const uint i = stride0 * did0 + stride1 * did1;
            dst[i + offset] *= alpha;
        }
    }
}

__kernel void ScaleTensor3d(global _FLOAT* __restrict dst,
                            const _FLOAT alpha,
                            const int offset,
                            const int stride0,
                            const int stride1,
                            const int stride2,
                            const int len0,
                            const int len1,
                            const int len2)
{
    const uint tidx = get_global_id(0);
    const uint tidy = get_global_id(1);
    const uint tidz = get_global_id(2);

    for(uint did0 = tidz; did0 < len0; did0 += GLOBAL_WORK_SIZE_Z)
    {
        for(uint did1 = tidy; did1 < len1; did1 += GLOBAL_WORK_SIZE_Y)
        {
            for(uint did2 = tidx; did2 < len2; did2 += GLOBAL_WORK_SIZE_X)
            {
                const uint i = stride0 * did0 + stride1 * did1 + stride2 * did2;

                dst[i + offset] *= alpha;
            }
        }
    }
}

__kernel void ScaleTensor4d(global _FLOAT* __restrict dst,
                            const _FLOAT alpha,
                            const int offset,
                            const int stride0,
                            const int stride1,
                            const int stride2,
                            const int stride3,
                            const int len0,
                            const int len1,
                            const int len2,
                            const int len3)
{
    const uint tidx = get_global_id(0);
    const uint tidy = get_global_id(1);
    const uint tidz = get_global_id(2);

    for(uint did0 = 0; did0 < len0; did0++)
    {
        for(uint did1 = tidz; did1 < len1; did1 += GLOBAL_WORK_SIZE_Z)
        {
            for(uint did2 = tidy; did2 < len2; did2 += GLOBAL_WORK_SIZE_Y)
            {
                for(uint did3 = tidx; did3 < len3; did3 += GLOBAL_WORK_SIZE_X)
                {
                    const uint i =
                        stride0 * did0 + stride1 * did1 + stride2 * did2 + stride3 * did3;

                    dst[i + offset] *= alpha;
                }
            }
        }
    }
}

__kernel void ScaleTensor5d(global _FLOAT* __restrict dst,
                            const _FLOAT alpha,
                            const int offset,
                            const int stride0,
                            const int stride1,
                            const int stride2,
                            const int stride3,
                            const int stride4,
                            const int len0,
                            const int len1,
                            const int len2,
                            const int len3,
                            const int len4)
{
    const uint tidx = get_global_id(0);
    const uint tidy = get_global_id(1);
    const uint tidz = get_global_id(2);

    for(uint did0 = 0; did0 < len0; did0++)
    {
        for(uint did1 = 0; did1 < len1; did1++)
        {
            for(uint did2 = tidz; did2 < len2; did2 += GLOBAL_WORK_SIZE_Z)
            {
                for(uint did3 = tidy; did3 < len3; did3 += GLOBAL_WORK_SIZE_Y)
                {
                    for(uint did4 = tidx; did4 < len4; did4 += GLOBAL_WORK_SIZE_X)
                    {
                        const uint i = stride0 * did0 + stride1 * did1 + stride2 * did2 +
                                       stride3 * did3 + stride4 * did4;

                        dst[i + offset] *= alpha;
                    }
                }
            }
        }
    }
}
