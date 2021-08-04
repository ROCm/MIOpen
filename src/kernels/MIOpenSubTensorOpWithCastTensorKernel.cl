/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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
#include "bfloat16_dev.hpp"

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_SRC_TYPE == 0
#define _FLOAT_SRC char
#elif MIOPEN_SRC_TYPE == 1
#define _FLOAT_SRC int
#elif MIOPEN_SRC_TYPE == 2
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT_SRC half
#elif MIOPEN_SRC_TYPE == 3
#define _FLOAT_SRC float
#else /* BFloat16 */
#define _FLOAT_SRC ushort
#endif

#if MIOPEN_DST_TYPE == 0
#define _FLOAT_DST char
#ifndef INT8_MAX
#define MAX_VAL 127 /* max value */
#else
#define MAX_VAL INT8_MAX
#endif
#elif MIOPEN_DST_TYPE == 1
#define _FLOAT_DST int
#ifndef INT32_MAX
#define MAX_VAL 2147483647 /* max value */
#else
#define MAX_VAL INT32_MAX
#endif
#elif MIOPEN_DST_TYPE == 2
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT_DST half
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#elif MIOPEN_DST_TYPE == 3
#define _FLOAT_DST float
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#else /* BFloat16 */
#define _FLOAT_DST ushort
#define MAX_VAL 0x7F7F /* max value */
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

__kernel void SubTensorOpWithCastTensor1d(const global _FLOAT_SRC* __restrict src,
                                          const float alpha,
                                          const int srcOffset,
                                          const int srcStride0,
                                          const int srcLen0,
                                          global _FLOAT_DST* __restrict dst,
                                          const int dstOffset,
                                          const int dstStride0)
{
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    for(uint did0 = did0_begin; did0 < srcLen0; did0 += WORK_LENGTH_0)
    {
        const uint sindex = srcStride0 * did0;
        const uint dindex = dstStride0 * did0;

        _FLOAT_SRC temp_src = *(src + sindex + srcOffset);
#if MIOPEN_SRC_TYPE == 3 && MIOPEN_DST_TYPE == 4
        temp_src *= alpha;
        *(dst + dindex + dstOffset) = float_to_bfloat16(temp_src);
#else
        bool over_flow              = (alpha * ((float)temp_src)) >= ((float)MAX_VAL);
        *(dst + dindex + dstOffset) = (_FLOAT_DST)(over_flow ? MAX_VAL : alpha * ((float)temp_src));
#endif
    }
}

__kernel void SubTensorOpWithCastTensor2d(const global _FLOAT_SRC* __restrict src,
                                          const float alpha,
                                          const int srcOffset,
                                          const int srcStride0,
                                          const int srcStride1,
                                          const int srcLen0,
                                          const int srcLen1,
                                          global _FLOAT_DST* __restrict dst,
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

            _FLOAT_SRC temp_src = *(src + sindex + srcOffset);
#if MIOPEN_SRC_TYPE == 3 && MIOPEN_DST_TYPE == 4
            temp_src *= alpha;
            *(dst + dindex + dstOffset) = float_to_bfloat16(temp_src);
#else
            bool over_flow = (alpha * ((float)temp_src)) >= ((float)MAX_VAL);
            *(dst + dindex + dstOffset) =
                (_FLOAT_DST)(over_flow ? MAX_VAL : alpha * ((float)temp_src));
#endif
        }
    }
}

__kernel void SubTensorOpWithCastTensor3d(const global _FLOAT_SRC* __restrict src,
                                          const float alpha,
                                          const int srcOffset,
                                          const int srcStride0,
                                          const int srcStride1,
                                          const int srcStride2,
                                          const int srcLen0,
                                          const int srcLen1,
                                          const int srcLen2,
                                          global _FLOAT_DST* __restrict dst,
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

                _FLOAT_SRC temp_src = *(src + sindex + srcOffset);

#if MIOPEN_SRC_TYPE == 3 && MIOPEN_DST_TYPE == 4
                temp_src *= alpha;
                *(dst + dindex + dstOffset) = float_to_bfloat16(temp_src);
#else
                bool over_flow = (alpha * ((float)temp_src)) >= ((float)MAX_VAL);
                *(dst + dindex + dstOffset) =
                    (_FLOAT_DST)(over_flow ? MAX_VAL : alpha * ((float)temp_src));
#endif
            }
        }
    }
}

__kernel void SubTensorOpWithCastTensor4d(const global _FLOAT_SRC* __restrict src,
                                          const float alpha,
                                          const int srcOffset,
                                          const int srcStride0,
                                          const int srcStride1,
                                          const int srcStride2,
                                          const int srcStride3,
                                          const int srcLen0,
                                          const int srcLen1,
                                          const int srcLen2,
                                          const int srcLen3,
                                          global _FLOAT_DST* __restrict dst,
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

                    _FLOAT_SRC temp_src = *(src + sindex + srcOffset);

#if MIOPEN_SRC_TYPE == 3 && MIOPEN_DST_TYPE == 4
                    temp_src *= alpha;
                    *(dst + dindex + dstOffset) = float_to_bfloat16(temp_src);
#else
                    bool over_flow = (alpha * ((float)temp_src)) >= ((float)MAX_VAL);
                    *(dst + dindex + dstOffset) =
                        (_FLOAT_DST)(over_flow ? MAX_VAL : alpha * ((float)temp_src));
#endif
                }
            }
        }
    }
}

__kernel void SubTensorOpWithCastTensor5d(const global _FLOAT_SRC* __restrict src,
                                          const float alpha,
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
                                          global _FLOAT_DST* __restrict dst,
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

                        _FLOAT_SRC temp_src = *(src + sindex + srcOffset);
#if MIOPEN_SRC_TYPE == 3 && MIOPEN_DST_TYPE == 4
                        temp_src *= alpha;
                        *(dst + dindex + dstOffset) = float_to_bfloat16(temp_src);
#else
                        bool over_flow = (alpha * ((float)temp_src)) >= ((float)MAX_VAL);
                        *(dst + dindex + dstOffset) =
                            (_FLOAT_DST)(over_flow ? MAX_VAL : alpha * ((float)temp_src));
#endif
                    }
                }
            }
        }
    }
}
