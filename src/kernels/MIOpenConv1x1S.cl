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

#include "float_types.h"
#include "math_ops.h"

// Since float_types.h has enabled true mixed precision for all
// direct ocl kernels, this kernel needs to retain its older behavior as it is
// dependent upon tunability which isn't slated for MIOpen 2.0 PR #1725
#if MIOPEN_USE_FP16 == 1
#define _FLOAT_PREC _FLOAT
#else
#define _FLOAT_PREC _FLOAT_ACCUM
#endif

#define UNUSED __attribute__((__unused__))

#define DBG_OUT_OF_RNGE 0

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenConv1x1(const __global _FLOAT* __restrict in_ptr,
              __constant _FLOAT* __restrict wei_ptr,
#if MLO_CONV_BIAS
              const __global _FLOAT* __restrict bias,
#endif
              __global _FLOAT* __restrict out_ptr,
              UNUSED _FLOAT dummy_val // nothing
)
{
    _FLOAT weights[MLO_N_LCL_OUT_MAPS][MLO_N_LCL_IN_MAPS];

    uint gbl_id0  = get_global_id(0);
    uint batch_id = gbl_id0 / MLO_MAP_SZ4; // batch
    uint pos      = gbl_id0 % MLO_MAP_SZ4;

    uint out_grp_block = get_group_id(1); // block of outputs for the entire group
    uint out_id        = out_grp_block * MLO_N_LCL_OUT_MAPS;

    uint gbl_in_off = batch_id * MLO_IN_BATCH_STRIDE + pos * MLO_READ_UNIT;

    uint wei_off = out_id *
#if MLO_DIR_FORWARD == 1
                   MLO_WEI_BSTRIDE
#else
                   MLO_WEI_CHANNEL_STRIDE
#endif
        ;

    _FLOAT_PREC accum[MLO_N_LCL_OUT_MAPS][MLO_READ_UNIT] = {{(_FLOAT_PREC)0.0f}};
    _FLOAT dat[MLO_N_LCL_IN_MAPS][MLO_READ_UNIT];

    for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
    {
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            accum[o][i] = (_FLOAT_PREC)0.0f;
        }
    }

    for(uint ci = 0; ci < MLO_CLOOP0; ++ci,
             // move input offset
                                      gbl_in_off += MLO_N_LCL_IN_MAPS * MLO_IN_CHANNEL_STRIDE,

             // move weights offset
                                      wei_off += MLO_N_LCL_IN_MAPS *
#if MLO_DIR_FORWARD == 1
                                                 MLO_WEI_CHANNEL_STRIDE
#else
                                                 MLO_WEI_BSTRIDE
#endif
    )
    {
        // read weights

        __constant _FLOAT* w1 = wei_ptr + wei_off;

        for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o,
                 w1 +=
#if MLO_DIR_FORWARD == 1
                                                MLO_WEI_BSTRIDE
#else
                                                MLO_WEI_CHANNEL_STRIDE
#endif
        )
        {
            __constant _FLOAT* w2 = w1;
            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c,
                     w2 +=
#if MLO_DIR_FORWARD == 1
                                                   MLO_WEI_CHANNEL_STRIDE
#else
                                                   MLO_WEI_BSTRIDE
#endif
            )
            {

                weights[o][c] = *w2;

#if DBG_OUT_OF_RNGE
                if(wei_off2 >= MLO_N_INPUTS * MLO_N_OUTPUTS)
                {
                    printf("K:oor: weights\n");
                }
#endif
            }
        }

        // convolve with all weights
        // read data
        // Shader compiler will use      GLOAL_LOAD_DWORD's OFFSET for *(ptr+index) access
        // Shader compiler will not use  GLOAL_LOAD_DWORD's OFFSET for ptr[index] access

        __global const _FLOAT* p = in_ptr + gbl_in_off;

        for(uint j = 0; j < MLO_N_LCL_IN_MAPS; j++)
        {
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                dat[j][i] = *(p + i);
#if DBG_OUT_OF_RNGE
                if(gbl_in_off1 + i >= MLO_IN_BATCH_STRIDE * MLO_BATCH_SZ)
                {
                    printf("K:oor: inputs\n");
                }
#endif
            }
            p += MLO_IN_CHANNEL_STRIDE;
        }

        // convolve
        for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
        {
            _FLOAT_PREC acc[MLO_READ_UNIT] = {(_FLOAT_PREC)0.0f};
            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                _FLOAT_PREC we = CVT_FLOAT2ACCUM(weights[o][c]);
                _FLOAT* d      = &dat[c][0];
                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    acc[i] += CVT_FLOAT2ACCUM(d[i]) * we;
                }
            }
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
                accum[o][i] += acc[i];
        }
    }

    uint gbl_out_off =
        batch_id * MLO_OUT_BATCH_STRIDE + pos * MLO_READ_UNIT + out_id * MLO_OUT_CHANNEL_STRIDE;
    __global _FLOAT* q = out_ptr + gbl_out_off;

    for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o, q += MLO_OUT_CHANNEL_STRIDE)
    {
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            *(q + i) = CVT_ACCUM2FLOAT(accum[o][i]);
        }
    }
}

/************************************************************************
stride and padding
*************************************************************************/
__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenConv1x1pquv(const __global _FLOAT* __restrict in_ptr,
                  __constant _FLOAT* __restrict wei_ptr,
#if MLO_CONV_BIAS
                  const __global _FLOAT* __restrict bias,
#endif
                  __global _FLOAT* __restrict out_ptr,
                  UNUSED _FLOAT dummy_val // nothing
)
{

    _FLOAT weights[MLO_N_LCL_OUT_MAPS][MLO_N_LCL_IN_MAPS];

    uint gbl_id0 = get_global_id(0);

    uint batch_id  = gbl_id0 / MLO_MAP_SZ4; // batch
    uint pos       = gbl_id0 % MLO_MAP_SZ4;
    uint pos_out_y = pos / MLO_OUT_WIDTH4;
    uint pos_out_x = pos % MLO_OUT_WIDTH4;

#if MLO_DIR_FORWARD == 1
    uint pos_in_y = pos_out_y * MLO_FILTER_STRIDE1;
    uint pos_in_x = pos_out_x * MLO_FILTER_STRIDE0;
#else
    uint pos_in_y = pos_out_y; /// MLO_FILTER_STRIDE1;   - divided already
    uint pos_in_x = pos_out_x; // MLO_FILTER_STRIDE0;  - divided already
#endif

    uint out_grp_block = get_group_id(1); // block of outputs for the entire group
    uint out_id        = out_grp_block * MLO_N_LCL_OUT_MAPS;

    uint gbl_in_off =
        batch_id * MLO_IN_BATCH_STRIDE + pos_in_y * MLO_IN_STRIDE + pos_in_x * MLO_READ_UNIT;
    //	bool vis = (pos_in_y < MLO_IN_HEIGHT);
    //	gbl_in_off = (vis) ? gbl_in_off : 0;

    uint wei_off = out_id *
#if MLO_DIR_FORWARD == 1
                   MLO_WEI_BSTRIDE
#else
                   MLO_WEI_CHANNEL_STRIDE
#endif
        ;

    _FLOAT_PREC accum[MLO_N_LCL_OUT_MAPS][MLO_READ_UNIT];
    _FLOAT dat[MLO_N_LCL_IN_MAPS][MLO_READ_UNIT];

    for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
    {
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            accum[o][i] = CVT_FLOAT2ACCUM(0);
        }
    }

    const __global _FLOAT* i_ptr = in_ptr + gbl_in_off;
    __constant _FLOAT* w_ptr     = wei_ptr + wei_off;
    for(uint ci = 0; ci < MLO_CLOOP0; ++ci)
    {

        // convolve with all weights
        // read data

        for(uint j = 0; j < MLO_N_LCL_IN_MAPS; ++j)
        {
            uint i = 0;
#if MLO_READ_UNIT > 1
            for(; i < MLO_READ_UNIT - 1; ++i)
            {
                uint off = i
#if MLO_DIR_FORWARD == 1
                           * MLO_FILTER_STRIDE0
#endif
                    ;
                dat[j][i] = *(i_ptr + off);
            }
#endif

            for(; i < MLO_READ_UNIT; ++i)
            {
                //				vis &= (pos_in_x + i*MLO_FILTER_STRIDE0 <
                // MLO_IN_WIDTH);
                uint off = i
#if MLO_DIR_FORWARD == 1
                           * MLO_FILTER_STRIDE0
#endif
                    ;
                //				off = (vis) ? off : 0;
                _FLOAT val = *(i_ptr + off);
                dat[j][i]  = val;
                //              dat[j][i] = (vis)? dat[j][i] : (_FLOAT)(0);
            }

            i_ptr += MLO_IN_CHANNEL_STRIDE;
        }
        // read weights
        __constant _FLOAT* w_ptr0 = w_ptr;

        for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
        {

            __constant _FLOAT* w_ptr1 = w_ptr0;

            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                weights[o][c] = *w_ptr1;
                w_ptr1 +=
#if MLO_DIR_FORWARD == 1
                    MLO_WEI_CHANNEL_STRIDE
#else
                    MLO_WEI_BSTRIDE
#endif
                    ;
            }

            w_ptr0 +=
#if MLO_DIR_FORWARD == 1
                MLO_WEI_BSTRIDE
#else
                MLO_WEI_CHANNEL_STRIDE
#endif
                ;
        }

        w_ptr += MLO_N_LCL_IN_MAPS *
#if MLO_DIR_FORWARD == 1
                 MLO_WEI_CHANNEL_STRIDE
#else
                 MLO_WEI_BSTRIDE
#endif
            ;
        // convolve
        for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
        {
            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    accum[o][i] += CVT_FLOAT2ACCUM(dat[c][i]) * CVT_FLOAT2ACCUM(weights[o][c]);
#if 0
                    if (pos_out_y == 2 && pos_out_x == 0)
                    {
                        printf((__constant char *)"K:c: %f %f %f %f\n",
                        accum[o][i],
                        dat[c][i] * weights[o][c],
                        dat[c][i],
                        weights[o][c]
                        );
                    }
#endif
                }
            }
        }
    }

    uint out_y = pos_out_y
#if MLO_DIR_FORWARD == 0
                 * MLO_FILTER_STRIDE1
#endif
        ;
    uint out_x = pos_out_x
#if MLO_DIR_FORWARD == 0
                 * MLO_FILTER_STRIDE0
#endif
        ;

    uint gbl_out_off = batch_id * MLO_OUT_BATCH_STRIDE + out_id * MLO_OUT_CHANNEL_STRIDE +
                       out_y * MLO_OUT_STRIDE + out_x * MLO_READ_UNIT;

    __global _FLOAT* q = out_ptr + gbl_out_off;

    for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o, q += MLO_OUT_CHANNEL_STRIDE)
    {

        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            __global _FLOAT* q1 = q;
            q1 += i
#if MLO_DIR_FORWARD == 0

                  * MLO_FILTER_STRIDE0
#endif
                ;
            *q1 = CVT_ACCUM2FLOAT(accum[o][i]);

#if MLO_DIR_FORWARD == 0
            for(uint s = 1; s < MLO_FILTER_STRIDE0; ++s)
            {
#if MLO_HORIZ_ALIGNED == 0
                if(out_x + s < MLO_OUT_WIDTH)
#endif
                {
                    *(q1 + s) = CVT_ACCUM2FLOAT(0);
                }
            }
#endif
        }

#if MLO_DIR_FORWARD == 0
        __global _FLOAT* q2 = q;
        for(uint j = 1; j < MLO_FILTER_STRIDE1; ++j)
        {
            q2 += MLO_OUT_STRIDE;
#if MLO_VERT_ALIGNED == 0
            if(out_y + j < MLO_OUT_HEIGHT)
#endif
            {

                for(uint s = 0; s < MLO_READ_UNIT * MLO_FILTER_STRIDE0; ++s)
                {
#if MLO_HORIZ_ALIGNED == 0
                    if(out_x + s < MLO_OUT_WIDTH)
#endif
                    {
                        *(q2 + s) = CVT_ACCUM2FLOAT(0);
                    }
                }
            }
        }
#endif
    }
}
