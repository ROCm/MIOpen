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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef MIOPEN_ALPHA_TYPE
#define MIOPEN_ALPHA_TYPE float
#endif

#ifndef MIOPEN_TYPE
#define MIOPEN_TYPE float
#endif

__kernel void SetTensor1d(global MIOPEN_TYPE* __restrict dst,
                          const MIOPEN_ALPHA_TYPE alpha,
                          const int offset,
                          const int stride0,
                          const int len0)
{
    const uint tidx = get_global_id(0);
    const uint tszx = get_global_size(0);

    for(uint did0 = tidx; did0 < len0; did0 += tszx)
    {
        const uint i = stride0 * did0;

        dst[i + offset] = alpha;
    }
}

__kernel void SetTensor2d(global MIOPEN_TYPE* __restrict dst,
                          const MIOPEN_ALPHA_TYPE alpha,
                          const int offset,
                          const int stride0,
                          const int stride1,
                          const int len0,
                          const int len1)
{
    const uint tidx = get_global_id(0);
    const uint tidy = get_global_id(1);

    const uint tszx = get_global_size(0);
    const uint tszy = get_global_size(1);

    for(uint did0 = tidy; did0 < len0; did0 += tszy)
    {
        for(uint did1 = tidx; did1 < len1; did1 += tszx)
        {
            const uint i = stride0 * did0 + stride1 * did1;

            dst[i + offset] = alpha;
        }
    }
}

__kernel void SetTensor3d(global MIOPEN_TYPE* __restrict dst,
                          const MIOPEN_ALPHA_TYPE alpha,
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

    const uint tszx = get_global_size(0);
    const uint tszy = get_global_size(1);
    const uint tszz = get_global_size(2);

    for(uint did0 = tidz; did0 < len0; did0 += tszz)
    {
        for(uint did1 = tidy; did1 < len1; did1 += tszy)
        {
            for(uint did2 = tidx; did2 < len2; did2 += tszx)
            {
                const uint i = stride0 * did0 + stride1 * did1 + stride2 * did2;

                dst[i + offset] = alpha;
            }
        }
    }
}

__kernel void SetTensor4d(global MIOPEN_TYPE* __restrict dst,
                          const MIOPEN_ALPHA_TYPE alpha,
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

    const uint tszx = get_global_size(0);
    const uint tszy = get_global_size(1);
    const uint tszz = get_global_size(2);

    for(uint did0 = 0; did0 < len0; did0++)
    {
        for(uint did1 = tidz; did1 < len1; did1 += tszz)
        {
            for(uint did2 = tidy; did2 < len2; did2 += tszy)
            {
                for(uint did3 = tidx; did3 < len3; did3 += tszx)
                {
                    const uint i =
                        stride0 * did0 + stride1 * did1 + stride2 * did2 + stride3 * did3;

                    dst[i + offset] = alpha;
                }
            }
        }
    }
}

__kernel void SetTensor5d(global MIOPEN_TYPE* __restrict dst,
                          const MIOPEN_ALPHA_TYPE alpha,
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

    const uint tszx = get_global_size(0);
    const uint tszy = get_global_size(1);
    const uint tszz = get_global_size(2);

    for(uint did0 = 0; did0 < len0; did0++)
    {
        for(uint did1 = 0; did1 < len1; did1++)
        {
            for(uint did2 = tidz; did2 < len2; did2 += tszz)
            {
                for(uint did3 = tidy; did3 < len3; did3 += tszy)
                {
                    for(uint did4 = tidx; did4 < len4; did4 += tszx)
                    {
                        const uint i = stride0 * did0 + stride1 * did1 + stride2 * did2 +
                                       stride3 * did3 + stride4 * did4;

                        dst[i + offset] = alpha;
                    }
                }
            }
        }
    }
}
