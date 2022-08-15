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

#include "float_types.h"

#if MIOPEN_USE_INT8 == 1 || MIOPEN_USE_INT8x4 == 1
#define _FLOAT char
#ifndef FLT_MAX
#define MAX_VAL 127 /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

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

#ifndef SUBTENSOR_OP_WITH_SCALAR
#define SUBTENSOR_OP_WITH_SCALAR BREAK_COMPILE_INTENTIONALLY
#endif

#define SUBTENSOR_OP_WITH_SCALAR_SET(t, a) (t = a)
#define SUBTENSOR_OP_WITH_SCALAR_MULTIPLY(t, a) (t *= a)

__kernel void SubTensorOpWithScalar1d(global _FLOAT* __restrict dst,
                                      const _FLOAT alpha,
                                      const int offset,
                                      const int stride0,
                                      const int len0)
{
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    for(uint did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        const uint i = stride0 * did0;

        SUBTENSOR_OP_WITH_SCALAR(dst[i + offset], alpha);
    }
}

__kernel void SubTensorOpWithScalar2d(global _FLOAT* __restrict dst,
                                      const _FLOAT alpha,
                                      const int offset,
                                      const int stride0,
                                      const int stride1,
                                      const int len0,
                                      const int len1)
{
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    itmp -= did0_begin * WORK_STRIDE_0;

    const uint did1_begin = itmp / WORK_STRIDE_1;

    for(uint did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(uint did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            const uint i = stride0 * did0 + stride1 * did1;

            SUBTENSOR_OP_WITH_SCALAR(dst[i + offset], alpha);
        }
    }
}

__kernel void SubTensorOpWithScalar3d(global _FLOAT* __restrict dst,
                                      const _FLOAT alpha,
                                      const int offset,
                                      const int stride0,
                                      const int stride1,
                                      const int stride2,
                                      const int len0,
                                      const int len1,
                                      const int len2)
{
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    itmp -= did0_begin * WORK_STRIDE_0;

    const uint did1_begin = itmp / WORK_STRIDE_1;

    itmp -= did1_begin * WORK_STRIDE_1;

    const uint did2_begin = itmp / WORK_STRIDE_2;

    for(uint did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(uint did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            for(uint did2 = did2_begin; did2 < len2; did2 += WORK_LENGTH_2)
            {
                const uint i = stride0 * did0 + stride1 * did1 + stride2 * did2;

                SUBTENSOR_OP_WITH_SCALAR(dst[i + offset], alpha);
            }
        }
    }
}

__kernel void SubTensorOpWithScalar4d(global _FLOAT* __restrict dst,
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
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    itmp -= did0_begin * WORK_STRIDE_0;

    const uint did1_begin = itmp / WORK_STRIDE_1;

    itmp -= did1_begin * WORK_STRIDE_1;

    const uint did2_begin = itmp / WORK_STRIDE_2;

    itmp -= did2_begin * WORK_STRIDE_2;

    const uint did3_begin = itmp / WORK_STRIDE_3;

    for(uint did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(uint did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            for(uint did2 = did2_begin; did2 < len2; did2 += WORK_LENGTH_2)
            {
                for(uint did3 = did3_begin; did3 < len3; did3 += WORK_LENGTH_3)
                {
                    const uint i =
                        stride0 * did0 + stride1 * did1 + stride2 * did2 + stride3 * did3;

                    SUBTENSOR_OP_WITH_SCALAR(dst[i + offset], alpha);
                }
            }
        }
    }
}

__kernel void SubTensorOpWithScalar5d(global _FLOAT* __restrict dst,
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

    for(uint did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(uint did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            for(uint did2 = did2_begin; did2 < len2; did2 += WORK_LENGTH_2)
            {
                for(uint did3 = did3_begin; did3 < len3; did3 += WORK_LENGTH_3)
                {
                    for(uint did4 = did4_begin; did4 < len4; did4 += WORK_LENGTH_4)
                    {
                        const uint i = stride0 * did0 + stride1 * did1 + stride2 * did2 +
                                       stride3 * did3 + stride4 * did4;

                        SUBTENSOR_OP_WITH_SCALAR(dst[i + offset], alpha);
                    }
                }
            }
        }
    }
}
