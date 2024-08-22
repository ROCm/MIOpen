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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
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

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#if MIOPEN_USE_INT8 == 1
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

#ifndef MIOPEN_BETA_IS_ZERO
#error "MIOPEN_BETA_IS_ZERO must be defined"
#endif
#ifndef MIOPEN_ALPHA_IS_ONE
#error "MIOPEN_ALPHA_IS_ONE must be defined"
#endif

#if MIOPEN_BETA_IS_ZERO && MIOPEN_ALPHA_IS_ONE
#define SUBTENSOR_OP_WITH_ALPHA_BETA(dst, src) \
    do                                         \
    {                                          \
        (dst) = (src);                         \
        (void)beta;                            \
        (void)alpha;                           \
    } while(0)
#elif MIOPEN_BETA_IS_ZERO
#define SUBTENSOR_OP_WITH_ALPHA_BETA(dst, src) \
    do                                         \
    {                                          \
        (dst) = (src)*alpha;                   \
        (void)beta;                            \
    } while(0)
#elif MIOPEN_ALPHA_IS_ONE
#define SUBTENSOR_OP_WITH_ALPHA_BETA(dst, src) \
    do                                         \
    {                                          \
        (dst) = mad((dst), beta, (src));       \
        (void)alpha;                           \
    } while(0)
#else
#define SUBTENSOR_OP_WITH_ALPHA_BETA(dst, src) \
    do                                         \
    {                                          \
        (dst) = mad((src), alpha, (dst)*beta); \
    } while(0)
#endif

__kernel void SubTensorOpWithTransform1d(global _FLOAT* __restrict src,
                                         const _FLOAT alpha,
                                         global _FLOAT* __restrict dst,
                                         const _FLOAT beta,
                                         const uint src_offset,
                                         const uint dst_offset,
                                         const uint src_stride0,
                                         const uint dst_stride0,
                                         const uint len0)
{
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    for(uint did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        uint si = src_stride0 * did0 + src_offset;
        uint di = dst_stride0 * did0 + dst_offset;

        SUBTENSOR_OP_WITH_ALPHA_BETA(dst[di], src[si]);
    }
}

__kernel void SubTensorOpWithTransform2d(global _FLOAT* __restrict src,
                                         const _FLOAT alpha,
                                         global _FLOAT* __restrict dst,
                                         const _FLOAT beta,
                                         const uint src_offset,
                                         const uint dst_offset,
                                         const uint src_stride0,
                                         const uint src_stride1,
                                         const uint dst_stride0,
                                         const uint dst_stride1,
                                         const uint len0,
                                         const uint len1)
{
    uint itmp = get_global_id(0);

    const uint did0_begin = itmp / WORK_STRIDE_0;

    itmp -= did0_begin * WORK_STRIDE_0;

    const uint did1_begin = itmp / WORK_STRIDE_1;

    for(uint did0 = did0_begin; did0 < len0; did0 += WORK_LENGTH_0)
    {
        for(uint did1 = did1_begin; did1 < len1; did1 += WORK_LENGTH_1)
        {
            uint si = src_stride0 * did0 + src_stride1 * did1 + src_offset;
            uint di = dst_stride0 * did0 + dst_stride1 * did1 + dst_offset;

            SUBTENSOR_OP_WITH_ALPHA_BETA(dst[di], src[si]);
        }
    }
}

__kernel void SubTensorOpWithTransform3d(global _FLOAT* __restrict src,
                                         const _FLOAT alpha,
                                         global _FLOAT* __restrict dst,
                                         const _FLOAT beta,
                                         const uint src_offset,
                                         const uint dst_offset,
                                         const uint src_stride0,
                                         const uint src_stride1,
                                         const uint src_stride2,
                                         const uint dst_stride0,
                                         const uint dst_stride1,
                                         const uint dst_stride2,
                                         const uint len0,
                                         const uint len1,
                                         const uint len2)
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
                uint si = src_stride0 * did0 + src_stride1 * did1 + src_stride2 * did2 + src_offset;
                uint di = dst_stride0 * did0 + dst_stride1 * did1 + dst_stride2 * did2 + dst_offset;

                SUBTENSOR_OP_WITH_ALPHA_BETA(dst[di], src[si]);
            }
        }
    }
}

__kernel void SubTensorOpWithTransform4d(global _FLOAT* __restrict src,
                                         const _FLOAT alpha,
                                         global _FLOAT* __restrict dst,
                                         const _FLOAT beta,
                                         const uint src_offset,
                                         const uint dst_offset,
                                         const uint src_stride0,
                                         const uint src_stride1,
                                         const uint src_stride2,
                                         const uint src_stride3,
                                         const uint dst_stride0,
                                         const uint dst_stride1,
                                         const uint dst_stride2,
                                         const uint dst_stride3,
                                         const uint len0,
                                         const uint len1,
                                         const uint len2,
                                         const uint len3)
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
                    uint si = src_stride0 * did0 + src_stride1 * did1 + src_stride2 * did2 +
                              src_stride3 * did3 + src_offset;
                    uint di = dst_stride0 * did0 + dst_stride1 * did1 + dst_stride2 * did2 +
                              dst_stride3 * did3 + dst_offset;

                    SUBTENSOR_OP_WITH_ALPHA_BETA(dst[di], src[si]);
                }
            }
        }
    }
}

__kernel void SubTensorOpWithTransform5d(global _FLOAT* __restrict src,
                                         const _FLOAT alpha,
                                         global _FLOAT* __restrict dst,
                                         const _FLOAT beta,
                                         const uint src_offset,
                                         const uint dst_offset,
                                         const uint src_stride0,
                                         const uint src_stride1,
                                         const uint src_stride2,
                                         const uint src_stride3,
                                         const uint src_stride4,
                                         const uint dst_stride0,
                                         const uint dst_stride1,
                                         const uint dst_stride2,
                                         const uint dst_stride3,
                                         const uint dst_stride4,
                                         const uint len0,
                                         const uint len1,
                                         const uint len2,
                                         const uint len3,
                                         const uint len4)
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
                        uint si = src_stride0 * did0 + src_stride1 * did1 + src_stride2 * did2 +
                                  src_stride3 * did3 + src_stride4 * did4 + src_offset;
                        uint di = dst_stride0 * did0 + dst_stride1 * did1 + dst_stride2 * did2 +
                                  dst_stride3 * did3 + dst_stride4 * did4 + dst_offset;

                        SUBTENSOR_OP_WITH_ALPHA_BETA(dst[di], src[si]);
                    }
                }
            }
        }
    }
}
