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

typedef struct
{
    long dims;
    long lens[5];
    long strides[5];
    long offset;
    long realsize;
} tensorDesc_t;

__kernel void CopyTensor(global MIOPEN_TYPE* __restrict src,
                         global MIOPEN_TYPE* __restrict dst,
                         tensorDesc_t srcDesc,
                         tensorDesc_t dstDesc)
{

    uint sindex = 0;
    uint dindex = 0;
    uint stmp, stmp2;
    uint dtmp, dtmp2;
    uint dims = srcDesc.dims;

    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);

    switch(dims)
    {
    case 1:
        if(gidx < dstDesc.realsize && gidx < srcDesc.realsize)
        {
            dst[gidx + dstDesc.offset] = src[gidx + srcDesc.offset];
        }
        break;
    case 2:
        if(gidx < dstDesc.lens[0] && gidy < dstDesc.lens[1])
        {
            dindex = dstDesc.strides[0] * gidx + gidy;
            sindex = srcDesc.strides[0] * gidx + gidy;
            if(dindex < dstDesc.realsize && sindex < srcDesc.realsize)
            {
                dst[dindex + dstDesc.offset] = src[sindex + srcDesc.offset];
            }
        }
        break;
    case 3:
        if(gidx < dstDesc.lens[0] && gidy < dstDesc.lens[1] && gidz < dstDesc.lens[2])
        {
            dindex = dstDesc.strides[0] * gidx + dstDesc.strides[1] * gidy + gidz;
            sindex = srcDesc.strides[0] * gidx + srcDesc.strides[1] * gidy + gidz;
            if(dindex < dstDesc.realsize && sindex < srcDesc.realsize)
            {
                dst[dindex + dstDesc.offset] = src[sindex + srcDesc.offset];
            }
        }
        break;
    case 4:
        if(gidx < dstDesc.lens[1] && gidy < dstDesc.lens[2] && gidz < dstDesc.lens[3])
        {
            stmp =
                srcDesc.strides[1] * gidx + srcDesc.strides[2] * gidy + srcDesc.strides[3] * gidz;
            dtmp =
                dstDesc.strides[1] * gidx + dstDesc.strides[2] * gidy + dstDesc.strides[3] * gidz;
            for(uint i = 0; i < srcDesc.lens[0]; i++)
            {
                sindex = stmp + srcDesc.strides[0] * i;
                dindex = dtmp + dstDesc.strides[0] * i;

                if(dindex < dstDesc.realsize && sindex < srcDesc.realsize)
                {
                    dst[dindex + dstDesc.offset] = src[sindex + srcDesc.offset];
                }
            }
        }
        break;
    case 5:
        if(gidx < dstDesc.lens[2] && gidy < dstDesc.lens[3] && gidz < dstDesc.lens[4])
        {
            stmp = srcDesc.strides[2] * gidx + srcDesc.strides[3] * gidy + gidz;
            dtmp = dstDesc.strides[2] * gidx + dstDesc.strides[3] * gidy + gidz;
            for(uint i = 0; i < srcDesc.lens[0]; i++)
            {
                stmp2 = stmp + srcDesc.strides[0] * i;
                dtmp2 = dtmp + dstDesc.strides[0] * i;

                for(uint j = 0; j < srcDesc.lens[1]; j++)
                {
                    sindex = stmp2 + srcDesc.strides[1] * j;
                    dindex = dtmp2 + dstDesc.strides[1] * j;

                    if(dindex < dstDesc.realsize && sindex < srcDesc.realsize)
                    {
                        dst[dindex + dstDesc.offset] = src[sindex + srcDesc.offset];
                    }
                }
            }
        }
        break;
    default: break;
    }
}
