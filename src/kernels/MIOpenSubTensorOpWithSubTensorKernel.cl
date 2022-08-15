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

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#ifndef MIOPEN_USE_BFP16
#define MIOPEN_USE_BFP16 0
#endif

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#ifndef MIOPEN_USE_INT8x4
#define MIOPEN_USE_INT8x4 0
#endif

#if MIOPEN_USE_INT8 == 1 || MIOPEN_USE_INT8x4 == 1
#define _FLOAT char
#ifndef FLT_MAX
#define MAX_VAL 127 /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif
#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_BFP16 == 1
#define _FLOAT ushort
#define MAX_VAL 0x7F7F /* max value */
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

#ifndef WORK_LENGTH_0
#define WORK_LENGTH_0 1
#endif

#ifndef WORK_LENGTH_1
#define WORK_LENGTH_1 1
#endif

#ifndef WORK_LENGTH_2
#define WORK_LENGTH_2 1
#endif

#ifndef WORK_LENGTH_3
#define WORK_LENGTH_3 1
#endif

#ifndef WORK_LENGTH_4
#define WORK_LENGTH_4 1
#endif

#define WORK_STRIDE_4 1
#define WORK_STRIDE_3 (WORK_LENGTH_4 * WORK_STRIDE_4)
#define WORK_STRIDE_2 (WORK_LENGTH_3 * WORK_STRIDE_3)
#define WORK_STRIDE_1 (WORK_LENGTH_2 * WORK_STRIDE_2)
#define WORK_STRIDE_0 (WORK_LENGTH_1 * WORK_STRIDE_1)

#ifndef SUBTENSOR_OP_WITH_SUBTENSOR
#define SUBTENSOR_OP_WITH_SUBTENSOR BREAK_COMPILE_INTENTIONALLY
#endif

#define SUBTENSOR_OP_WITH_SUBTENSOR_COPY(dst, src) (dst = src)

__kernel void SubTensorOpWithSubTensor1d(const global _FLOAT* __restrict src,
                                         const int srcOffset,
                                         const int srcStride0,
                                         const int srcLen0,
                                         global _FLOAT* __restrict dst,
                                         const int dstOffset,
                                         const int dstStride0)
{
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    for(uint did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        const uint sindex = srcStride0 * did0;
        const uint dindex = dstStride0 * did0;

        SUBTENSOR_OP_WITH_SUBTENSOR(dst[dindex + dstOffset], src[sindex + srcOffset]);
    }
}

__kernel void SubTensorOpWithSubTensor2d(const global _FLOAT* __restrict src,
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
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    itmp -= did0_begin * WORK_STRIDE_0;

    const uint did1_begin = itmp / WORK_STRIDE_1;

    for(uint did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(uint did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            const uint sindex = srcStride0 * did0 + srcStride1 * did1;
            const uint dindex = dstStride0 * did0 + dstStride1 * did1;

            SUBTENSOR_OP_WITH_SUBTENSOR(dst[dindex + dstOffset], src[sindex + srcOffset]);
        }
    }
}

__kernel void SubTensorOpWithSubTensor3d(const global _FLOAT* __restrict src,
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
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    itmp -= did0_begin * WORK_STRIDE_0;

    const uint did1_begin = itmp / WORK_STRIDE_1;

    itmp -= did1_begin * WORK_STRIDE_1;

    const uint did2_begin = itmp / WORK_STRIDE_2;

    for(uint did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(uint did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            for(uint did2 = did2_begin; did2 < srcLen2; did2 += WORK_LENGTH_2)
            {
                const uint sindex = srcStride0 * did0 + srcStride1 * did1 + srcStride2 * did2;
                const uint dindex = dstStride0 * did0 + dstStride1 * did1 + dstStride2 * did2;

                SUBTENSOR_OP_WITH_SUBTENSOR(dst[dindex + dstOffset], src[sindex + srcOffset]);
            }
        }
    }
}

__kernel void SubTensorOpWithSubTensor4d(const global _FLOAT* __restrict src,
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
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    itmp -= did0_begin * WORK_STRIDE_0;

    const uint did1_begin = itmp / WORK_STRIDE_1;

    itmp -= did1_begin * WORK_STRIDE_1;

    const uint did2_begin = itmp / WORK_STRIDE_2;

    itmp -= did2_begin * WORK_STRIDE_2;

    const uint did3_begin = itmp / WORK_STRIDE_3;

    for(uint did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(uint did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            for(uint did2 = did2_begin; did2 < srcLen2; did2 += WORK_LENGTH_2)
            {
                for(uint did3 = did3_begin; did3 < srcLen3; did3 += WORK_LENGTH_3)
                {
                    const uint sindex = srcStride0 * did0 + srcStride1 * did1 + srcStride2 * did2 +
                                        srcStride3 * did3;
                    const uint dindex = dstStride0 * did0 + dstStride1 * did1 + dstStride2 * did2 +
                                        dstStride3 * did3;

                    SUBTENSOR_OP_WITH_SUBTENSOR(dst[dindex + dstOffset], src[sindex + srcOffset]);
                }
            }
        }
    }
}

__kernel void SubTensorOpWithSubTensor5d(const global _FLOAT* __restrict src,
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
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    itmp -= did0_begin * WORK_STRIDE_0;

    const uint did1_begin = itmp / WORK_STRIDE_1;

    itmp -= did1_begin * WORK_STRIDE_1;

    const uint did2_begin = itmp / WORK_STRIDE_2;

    itmp -= did2_begin * WORK_STRIDE_2;

    const uint did3_begin = itmp / WORK_STRIDE_3;

    itmp -= did3_begin * WORK_STRIDE_3;

    const uint did4_begin = itmp / WORK_STRIDE_4;

    for(uint did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        for(uint did1 = did1_begin; did1 < srcLen1; did1 += WORK_LENGTH_1)
        {
            for(uint did2 = did2_begin; did2 < srcLen2; did2 += WORK_LENGTH_2)
            {
                for(uint did3 = did3_begin; did3 < srcLen3; did3 += WORK_LENGTH_3)
                {
                    for(uint did4 = did4_begin; did4 < srcLen4; did4 += WORK_LENGTH_4)
                    {
                        const uint sindex = srcStride0 * did0 + srcStride1 * did1 +
                                            srcStride2 * did2 + srcStride3 * did3 +
                                            srcStride4 * did4;
                        const uint dindex = dstStride0 * did0 + dstStride1 * did1 +
                                            dstStride2 * did2 + dstStride3 * did3 +
                                            dstStride4 * did4;

                        SUBTENSOR_OP_WITH_SUBTENSOR(dst[dindex + dstOffset],
                                                    src[sindex + srcOffset]);
                    }
                }
            }
        }
    }
}
