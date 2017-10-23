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



typedef struct {
  long dims;
  long lens[5];
  long strides[5];
  long offset;
  long size;
}tensorDesc_t;


__kernel void CopyTensor(global MIOPEN_TYPE* __restrict src, 
                         global MIOPEN_TYPE* __restrict dst, tensorDesc_t srcDesc, tensorDesc_t dstDesc)
{
             
    uint sindex = 0;
    uint dindex = 0;
    uint dims   = srcDesc.dims;
    
    uint gidx = get_global_id(0);
    uint gidy = get_global_id(1);
    uint gidz = get_global_id(2);
    
    switch(dims)
    {
        case 1:
            if(gidx < dstDesc.size && gidx < srcDesc.size)
            {   
                dst[gidx] = src[gidx];
            }
            break;
        case 2:
            dindex = dstDesc.strides[0]*gidx + gidy;
            sindex = srcDesc.strides[0]*gidx + gidy;
            if(dindex < dstDesc.size && sindex < srcDesc.size)
            {   
                dst[dindex] = src[sindex];
            }
            break;
        case 3:
            dindex = dstDesc.strides[0]*gidx + dstDesc.strides[1]*gidy + gidz;
            sindex = srcDesc.strides[0]*gidx + srcDesc.strides[1]*gidy + gidz;
            if(dindex < dstDesc.size && sindex < srcDesc.size)
            {   
                dst[dindex] = src[sindex];
            }
            break;
        case 4:
            sindex = srcDesc.strides[1]*gidx + srcDesc.strides[2]*gidy + srcDesc.strides[3]*gidz;
            dindex = dstDesc.strides[1]*gidx + dstDesc.strides[2]*gidy + dstDesc.strides[3]*gidz;
            for(uint i = 0; i < srcDesc.lens[0]; i++)
            {
                sindex += srcDesc.strides[0]*i;
                dindex += dstDesc.strides[0]*i;

                if(dindex < dstDesc.size && sindex < srcDesc.size)
                {  
                   dst[dindex] = src[sindex];
                }

            }
            break;
        case 5:
            sindex = srcDesc.strides[2]*gidx + srcDesc.strides[3]*gidy + gidz;
            dindex = dstDesc.strides[2]*gidx + dstDesc.strides[3]*gidy + gidz;
            for(uint i = 0; i < srcDesc.lens[0]; i++)
            {
                sindex += srcDesc.strides[0]*i;
                dindex += dstDesc.strides[0]*i; 
                
                for(uint j = 0; j < srcDesc.lens[1]; j++)
                {
                    sindex += srcDesc.strides[1]*j;
                    dindex += dstDesc.strides[1]*j; 

                    if(dindex < dstDesc.size && sindex < srcDesc.size)
                    {  
                       dst[dindex] = src[sindex];
                    }
                }
            }
            break;
        default:
            break;
    }
}

    

