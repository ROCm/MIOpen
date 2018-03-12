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

__kernel void CopyTensor1d(const global _FLOAT* __restrict src,
                           const int srcOffset,
                           const int srcStride0,
                           const int srcLen0,
                           global _FLOAT* __restrict dst,
                           const int dstOffset,
                           const int dstStride0)
{
    const uint tidx = get_global_id(0);

    for(uint did0 = tidx; did0 < srcLen0; did0 += GLOBAL_WORK_SIZE_X)
    {
        const uint sindex = srcStride0 * did0;
        const uint dindex = dstStride0 * did0;

        dst[dindex + dstOffset] = src[sindex + srcOffset];
    }
}

__kernel void CopyTensor2d(const global _FLOAT* __restrict src,
                           const int srcOffset,
                           const int srcStride0,
                           const int srcStride1,
                           const int srcLen0,
                           const int srcLen1,
                           global _FLOAT* __restrict dst,
                           const int dstOffset,
                           const int dstStride0,
                           const int dstStride1)
{
    const uint tidx = get_global_id(0);
    const uint tidy = get_global_id(1);

    for(uint did0 = tidy; did0 < srcLen0; did0 += GLOBAL_WORK_SIZE_Y)
    {
        for(uint did1 = tidx; did1 < srcLen1; did1 += GLOBAL_WORK_SIZE_X)
        {
            const uint sindex = srcStride0 * did0 + srcStride1 * did1;
            const uint dindex = dstStride0 * did0 + dstStride1 * did1;

            dst[dindex + dstOffset] = src[sindex + srcOffset];
        }
    }
}

__kernel void CopyTensor3d(const global _FLOAT* __restrict src,
                           const int srcOffset,
                           const int srcStride0,
                           const int srcStride1,
                           const int srcStride2,
                           const int srcLen0,
                           const int srcLen1,
                           const int srcLen2,
                           global _FLOAT* __restrict dst,
                           const int dstOffset,
                           const int dstStride0,
                           const int dstStride1,
                           const int dstStride2)
{
    const uint tidx = get_global_id(0);
    const uint tidy = get_global_id(1);
    const uint tidz = get_global_id(2);

    for(uint did0 = tidz; did0 < srcLen0; did0 += GLOBAL_WORK_SIZE_Z)
    {
        for(uint did1 = tidy; did1 < srcLen1; did1 += GLOBAL_WORK_SIZE_Y)
        {
            for(uint did2 = tidx; did2 < srcLen2; did2 += GLOBAL_WORK_SIZE_X)
            {
                const uint sindex = srcStride0 * did0 + srcStride1 * did1 + srcStride2 * did2;
                const uint dindex = dstStride0 * did0 + dstStride1 * did1 + dstStride2 * did2;

                dst[dindex + dstOffset] = src[sindex + srcOffset];
            }
        }
    }
}

__kernel void CopyTensor4d(const global _FLOAT* __restrict src,
                           const int srcOffset,
                           const int srcStride0,
                           const int srcStride1,
                           const int srcStride2,
                           const int srcStride3,
                           const int srcLen0,
                           const int srcLen1,
                           const int srcLen2,
                           const int srcLen3,
                           global _FLOAT* __restrict dst,
                           const int dstOffset,
                           const int dstStride0,
                           const int dstStride1,
                           const int dstStride2,
                           const int dstStride3)
{
    const uint tidx = get_global_id(0);
    const uint tidy = get_global_id(1);
    const uint tidz = get_global_id(2);

    for(uint did0 = 0; did0 < srcLen0; did0++)
    {
        for(uint did1 = tidz; did1 < srcLen1; did1 += GLOBAL_WORK_SIZE_Z)
        {
            for(uint did2 = tidy; did2 < srcLen2; did2 += GLOBAL_WORK_SIZE_Y)
            {
                for(uint did3 = tidx; did3 < srcLen3; did3 += GLOBAL_WORK_SIZE_X)
                {
                    const uint sindex = srcStride0 * did0 + srcStride1 * did1 + srcStride2 * did2 +
                                        srcStride3 * did3;
                    const uint dindex = dstStride0 * did0 + dstStride1 * did1 + dstStride2 * did2 +
                                        dstStride3 * did3;

                    dst[dindex + dstOffset] = src[sindex + srcOffset];
                }
            }
        }
    }
}

__kernel void CopyTensor5d(const global _FLOAT* __restrict src,
                           const int srcOffset,
                           const int srcStride0,
                           const int srcStride1,
                           const int srcStride2,
                           const int srcStride3,
                           const int srcStride4,
                           const int srcLen0,
                           const int srcLen1,
                           const int srcLen2,
                           const int srcLen3,
                           const int srcLen4,
                           global _FLOAT* __restrict dst,
                           const int dstOffset,
                           const int dstStride0,
                           const int dstStride1,
                           const int dstStride2,
                           const int dstStride3,
                           const int dstStride4)
{
    const uint tidx = get_global_id(0);
    const uint tidy = get_global_id(1);
    const uint tidz = get_global_id(2);

    for(uint did0 = 0; did0 < srcLen0; did0++)
    {
        for(uint did1 = 0; did1 < srcLen1; did1++)
        {
            for(uint did2 = tidz; did2 < srcLen2; did2 += GLOBAL_WORK_SIZE_Z)
            {
                for(uint did3 = tidy; did3 < srcLen3; did3 += GLOBAL_WORK_SIZE_Y)
                {
                    for(uint did4 = tidx; did4 < srcLen4; did4 += GLOBAL_WORK_SIZE_X)
                    {
                        const uint sindex = srcStride0 * did0 + srcStride1 * did1 +
                                            srcStride2 * did2 + srcStride3 * did3 +
                                            srcStride4 * did4;
                        const uint dindex = dstStride0 * did0 + dstStride1 * did1 +
                                            dstStride2 * did2 + dstStride3 * did3 +
                                            dstStride4 * did4;

                        dst[dindex + dstOffset] = src[sindex + srcOffset];
                    }
                }
            }
        }
    }
}
