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
__kernel void ScaleTensor(global MIOPEN_TYPE* __restrict dst,
                          MIOPEN_ALPHA_TYPE alpha,
                          long num_elems,
                          long offset)
{
    uint gid = get_global_id(0);
    if(gid < num_elems)
    {
        dst[gid + offset] *= alpha;
    }
}

__kernel void
SetTensor(global MIOPEN_TYPE* __restrict dst, MIOPEN_ALPHA_TYPE alpha, long num_elems, long offset)
{
    uint gid = get_global_id(0);
    if(gid < num_elems)
    {
        dst[gid + offset] = alpha;
    }
}

#ifndef MIO_TC_USE_COPYKERNEL
#define MIO_TC_USE_COPYKERNEL 0
#endif

#if(MIO_TC_USE_COPYKERNEL == 1)

#ifndef MIO_TC_DIMS
#define MIO_TC_DIMS 3
#endif

__kernel void CopyTensor(global MIOPEN_TYPE* __restrict src,
                         global MIOPEN_TYPE* __restrict dst,
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
                         const long srcRealsize,
                         const int dstOffset,
                         const int dstStride0,
                         const int dstStride1,
                         const int dstStride2,
                         const int dstStride3,
                         const int dstStride4,
                         const int dstLen0,
                         const int dstLen1,
                         const int dstLen2,
                         const int dstLen3,
                         const int dstLen4,
                         const long dstRealsize)
{
#if(MIO_TC_DIMS > 1)
    uint sindex = 0;
    uint dindex = 0;
#endif

#if(MIO_TC_DIMS == 1)
    uint gidx = get_global_id(0);
    if(gidx < dstRealsize && gidx < srcRealsize)
    {
        dst[gidx + dstOffset] = src[gidx + srcOffset];
    }
#elif(MIO_TC_DIMS == 2)
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    if(gidx < dstLen0 && gidy < dstLen1)
    {
        dindex = dstStride0 * gidx + gidy;
        sindex = srcStride0 * gidx + gidy;
        if(dindex < dstRealsize && sindex < srcRealsize)
        {
            dst[dindex + dstOffset] = src[sindex + srcOffset];
        }
    }
#elif(MIO_TC_DIMS == 3)
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);
    if(gidx < dstLen0 && gidy < dstLen1 && gidz < dstLen2)
    {
        dindex = dstStride0 * gidx + dstStride1 * gidy + gidz;
        sindex = srcStride0 * gidx + srcStride1 * gidy + gidz;
        if(dindex < dstRealsize && sindex < srcRealsize)
        {
            dst[dindex + dstOffset] = src[sindex + srcOffset];
        }
    }
#elif(MIO_TC_DIMS == 4)
    uint stmp;
    uint dtmp;
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);
    if(gidx < dstLen1 && gidy < dstLen2 && gidz < dstLen3)
    {
        stmp = srcStride1 * gidx + srcStride2 * gidy + srcStride3 * gidz;
        dtmp = dstStride1 * gidx + dstStride2 * gidy + dstStride3 * gidz;
#pragma unroll
        for(uint i = 0; i < srcLen0; i++)
        {
            sindex = stmp + srcStride0 * i;
            dindex = dtmp + dstStride0 * i;

            if(dindex < dstRealsize && sindex < srcRealsize)
            {
                dst[dindex + dstOffset] = src[sindex + srcOffset];
            }
        }
    }
#elif(MIO_TC_DIMS == 5)
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);
    uint stmp, stmp2;
    uint dtmp, dtmp2;
    if(gidx < dstLen2 && gidy < dstLen3 && gidz < dstLen4)
    {
        stmp = srcStride2 * gidx + srcStride3 * gidy + gidz;
        dtmp = dstStride2 * gidx + dstStride3 * gidy + gidz;
#pragma unroll
        for(uint i = 0; i < srcLen0; i++)
        {
            stmp2 = stmp + srcStride0 * i;
            dtmp2 = dtmp + dstStride0 * i;
#pragma unroll
            for(uint j = 0; j < srcLen1; j++)
            {
                sindex = stmp2 + srcStride1 * j;
                dindex = dtmp2 + dstStride1 * j;
                if(dindex < dstRealsize && sindex < srcRealsize)
                {
                    dst[dindex + dstOffset] = src[sindex + srcOffset];
                }
            }
        }
    }
#endif
}

#endif
