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

#ifndef MIOPEN_ALPHA_TYPE
#define MIOPEN_ALPHA_TYPE float
#endif

#ifndef MIOPEN_TYPE
#define MIOPEN_TYPE float
#endif

__kernel void
ScaleTensor(global MIOPEN_TYPE* __restrict dst, MIOPEN_ALPHA_TYPE alpha, long num_elems)
{
    uint gid = get_global_id(0);
    if(gid < num_elems)
    {
        dst[gid] *= alpha;
    }
}

__kernel void SetTensor(global MIOPEN_TYPE* __restrict dst, MIOPEN_ALPHA_TYPE alpha, long num_elems)
{
    uint gid = get_global_id(0);
    if(gid < num_elems)
    {
        dst[gid] = alpha;
    }
}

__kernel void CopyTensor(global MIOPEN_TYPE* __restrict src,
                         global MIOPEN_TYPE* __restrict dst,
                         const int srcOffset,
                         const int srcStride0,
                         const int srcStride1,
                         const int srcStride2,
                         const int srcStride3,
                         const int srcLen0,
                         const int srcLen1,
                         const long srcRealsize,
                         const int dstOffset,
                         const int dstStride0,
                         const int dstStride1,
                         const int dstStride2,
                         const int dstStride3,
                         const int dstLen0,
                         const int dstLen1,
                         const int dstLen2,
                         const int dstLen3,
                         const int dstLen4,
                         const long dstRealsize,
                         const int dims)
{

    uint sindex = 0;
    uint dindex = 0;
    uint gidx   = get_global_id(0);
    uint gidy   = get_global_id(1);
    uint gidz   = get_global_id(2);
    uint stmp, stmp2;
    uint dtmp, dtmp2;

    switch(dims)
    {
    case 1:
        for(int idx = gidx; idx < dstLen0; idx += 65536)
        {
            if(idx < dstRealsize && idx < srcRealsize)
            {
                dst[idx + dstOffset] = src[idx + srcOffset];
            }
        }
        break;

    case 2:
        for(int xidx = gidx; xidx < dstLen0; xidx += 256)
        {
            for(int yidx = gidy; yidx < dstLen1; yidx += 256)
            {
                dindex = dstStride0 * xidx + yidx;
                sindex = srcStride0 * xidx + yidx;
                if(dindex < dstRealsize && sindex < srcRealsize)
                {
                    dst[dindex + dstOffset] = src[sindex + srcOffset];
                }
            }
        }
        break;

    case 3:

        for(int xidx = gidx; xidx < dstLen0; xidx += 16)
        {
            for(int yidx = gidy; yidx < dstLen1; yidx += 64)
            {
                for(int zidx = gidz; zidx < dstLen2; zidx += 64)
                {

                    dindex = dstStride0 * xidx + dstStride1 * yidx + zidx;
                    sindex = srcStride0 * xidx + srcStride1 * yidx + zidx;

                    if(dindex < dstRealsize && sindex < srcRealsize)
                    {
                        dst[dindex + dstOffset] = src[sindex + srcOffset];
                    }
                }
            }
        }
        break;

    case 4:

        for(int xidx = gidx; xidx < dstLen1; xidx += 16)
        {
            for(int yidx = gidy; yidx < dstLen2; yidx += 64)
            {
                for(int zidx = gidz; zidx < dstLen3; zidx += 64)
                {
                    stmp = srcStride1 * xidx + srcStride2 * yidx + srcStride3 * zidx;
                    dtmp = dstStride1 * xidx + dstStride2 * yidx + dstStride3 * zidx;
#pragma unroll
                    for(int idx = 0; idx < srcLen0; idx++)
                    {
                        sindex = stmp + srcStride0 * idx;
                        dindex = dtmp + dstStride0 * idx;

                        if(dindex < dstRealsize && sindex < srcRealsize)
                        {
                            dst[dindex + dstOffset] = src[sindex + srcOffset];
                        }
                    }
                }
            }
        }
        break;

    case 5:
        for(int xidx = gidx; xidx < dstLen2; xidx += 16)
        {
            for(int yidx = gidy; yidx < dstLen3; yidx += 64)
            {
                for(int zidx = gidz; zidx < dstLen4; zidx += 64)
                {
                    stmp = srcStride2 * xidx + srcStride3 * yidx + zidx;
                    dtmp = dstStride2 * xidx + dstStride3 * yidx + zidx;

#pragma unroll
                    for(int idx = 0; idx < srcLen0; idx++)
                    {
                        stmp2 = stmp + srcStride0 * idx;
                        dtmp2 = dtmp + dstStride0 * idx;
#pragma unroll
                        for(int jdx = 0; jdx < srcLen1; jdx++)
                        {
                            sindex = stmp2 + srcStride1 * jdx;
                            dindex = dtmp2 + dstStride1 * jdx;
                            if(dindex < dstRealsize && sindex < srcRealsize)
                            {
                                dst[dindex + dstOffset] = src[sindex + srcOffset];
                            }
                        }
                    }
                }
            }
        }
        break;

    default: break;
    }
}
