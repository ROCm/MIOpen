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

#define _FLOAT float
#define _FLOAT2 float2
#define _FLOAT4 float4
#define _FLOAT8 float8

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F /* max value */
#endif

#define UNUSED __attribute__((__unused__))

#define DBG_OUT_OF_RNGE 0

// calculating the size of the area for weights prefetch

__attribute__((always_inline)) uint iDiv(uint v, uint d)
{
    uint r = (uint)((float)v * (1.0f / (float)d) + 0.00001f);
    return (r);
}

__attribute__((always_inline)) uint iMod(uint v, uint u, uint d)
{
    uint r = v - mul24(u, d);
    return (r);
}

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
    uint batch_id = iDiv(gbl_id0, MLO_MAP_SZ4); // batch
    uint pos      = iMod(gbl_id0, batch_id, MLO_MAP_SZ4);

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

    _FLOAT accum[MLO_N_LCL_OUT_MAPS][MLO_READ_UNIT];
    _FLOAT dat[MLO_N_LCL_IN_MAPS][MLO_READ_UNIT];

    for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
    {
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            accum[o][i] = 0;
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

        for(uint o = 0, wei_off1 = wei_off; o < MLO_N_LCL_OUT_MAPS; ++o,
                 wei_off1 +=
#if MLO_DIR_FORWARD == 1
                                                                    MLO_WEI_BSTRIDE
#else
                                                                    MLO_WEI_CHANNEL_STRIDE
#endif
            )
        {
            for(uint c = 0, wei_off2 = wei_off1; c < MLO_N_LCL_IN_MAPS; ++c,
                     wei_off2 +=
#if MLO_DIR_FORWARD == 1
                                                                        MLO_WEI_CHANNEL_STRIDE
#else
                                                                        MLO_WEI_BSTRIDE
#endif
                )
            {
                weights[o][c] = wei_ptr[wei_off2];
            }
        }

        // convolve with all weights
        // read data
        for(uint j = 0, gbl_in_off1 = gbl_in_off; j < MLO_N_LCL_IN_MAPS;
            ++j, gbl_in_off1 += MLO_IN_CHANNEL_STRIDE)
        {
            for(uint i = 0; i < MLO_READ_UNIT; ++i)
            {
                dat[j][i] = in_ptr[gbl_in_off1 + i];
            }
        }

        // convolve
        for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
        {
            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    accum[o][i] += dat[c][i] * weights[o][c];
                }
            }
        }
    }

    uint gbl_out_off =
        batch_id * MLO_OUT_BATCH_STRIDE + pos * MLO_READ_UNIT + out_id * MLO_OUT_CHANNEL_STRIDE;
    for(uint o = 0, gbl_out_off1 = gbl_out_off; o < MLO_N_LCL_OUT_MAPS;
        ++o, gbl_out_off1 += MLO_OUT_CHANNEL_STRIDE)
    {
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            out_ptr[gbl_out_off1 + i] = accum[o][i];
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

    uint gbl_id0  = get_global_id(0);

    uint batch_id = iDiv(gbl_id0, MLO_MAP_SZ4); // batch
    uint pos      = iMod(gbl_id0, batch_id, MLO_MAP_SZ4);
	uint pos_out_y    = iDiv(pos, MLO_OUT_WIDTH4);
	uint pos_out_x    = iMod(pos, pos_out_y, MLO_OUT_WIDTH4);

#if MLO_DIR_FORWARD == 1
	uint pos_in_y = pos_out_y*MLO_FILTER_STRIDE1;
	uint pos_in_x = pos_out_x*MLO_FILTER_STRIDE0;
#else
	uint pos_in_y = pos_out_y; ///MLO_FILTER_STRIDE1;   - divided already
	uint pos_in_x = pos_out_x; //MLO_FILTER_STRIDE0;  - divided already
#endif

    uint out_grp_block = get_group_id(1); // block of outputs for the entire group
    uint out_id        = out_grp_block * MLO_N_LCL_OUT_MAPS;

    uint gbl_in_off = batch_id * MLO_IN_BATCH_STRIDE + pos_in_y*MLO_IN_STRIDE + pos_in_x * MLO_READ_UNIT;
//	bool vis = (pos_in_y < MLO_IN_HEIGHT);
//	gbl_in_off = (vis) ? gbl_in_off : 0;

    uint wei_off = out_id *
#if MLO_DIR_FORWARD == 1
                   MLO_WEI_BSTRIDE
#else
                   MLO_WEI_CHANNEL_STRIDE
#endif
        ;

    _FLOAT accum[MLO_N_LCL_OUT_MAPS][MLO_READ_UNIT];
    _FLOAT dat[MLO_N_LCL_IN_MAPS][MLO_READ_UNIT];

    for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
    {
        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
            accum[o][i] = 0;
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

        for(uint o = 0, wei_off1 = wei_off; o < MLO_N_LCL_OUT_MAPS; ++o,
                 wei_off1 +=
#if MLO_DIR_FORWARD == 1
                                                                    MLO_WEI_BSTRIDE
#else
                                                                    MLO_WEI_CHANNEL_STRIDE
#endif
            )
        {
            for(uint c = 0, wei_off2 = wei_off1; c < MLO_N_LCL_IN_MAPS; ++c,
                     wei_off2 +=
#if MLO_DIR_FORWARD == 1
                                                                        MLO_WEI_CHANNEL_STRIDE
#else
                                                                        MLO_WEI_BSTRIDE
#endif
                )
            {
                weights[o][c] = wei_ptr[wei_off2];
            }
        }

        // convolve with all weights
        // read data
        for(uint j = 0, gbl_in_off1 = gbl_in_off; j < MLO_N_LCL_IN_MAPS;
            ++j, gbl_in_off1 += MLO_IN_CHANNEL_STRIDE)
        {
			uint i = 0;
#if MLO_READ_UNIT > 1
            for(; i < MLO_READ_UNIT - 1; ++i)
            {
				uint off = gbl_in_off1 + i
#if MLO_DIR_FORWARD == 1
				*MLO_FILTER_STRIDE0
#endif
				;
                dat[j][i] = in_ptr[off];
            }
#endif

            for(; i < MLO_READ_UNIT; ++i)
            {
//				vis &= (pos_in_x + i*MLO_FILTER_STRIDE0 < MLO_IN_WIDTH);
				uint off = gbl_in_off1 + i
#if MLO_DIR_FORWARD == 1
				*MLO_FILTER_STRIDE0
#endif
				;
//				off = (vis) ? off : 0;
				_FLOAT val = in_ptr[off];
                dat[j][i] = val;
//              dat[j][i] = (vis)? dat[j][i] : 0;

            }

        }

        // convolve
        for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
        {
            for(uint c = 0; c < MLO_N_LCL_IN_MAPS; ++c)
            {
                for(uint i = 0; i < MLO_READ_UNIT; ++i)
                {
                    accum[o][i] += dat[c][i] * weights[o][c];
#if 0
				if (get_global_id(0) == 7 && get_global_id(1) ==0 && o == 0 && i == 0)
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
	uint out_x =  pos_out_x
#if MLO_DIR_FORWARD == 0
        * MLO_FILTER_STRIDE0
#endif
	;

    uint gbl_out_off =
        batch_id * MLO_OUT_BATCH_STRIDE + out_id * MLO_OUT_CHANNEL_STRIDE + out_y * MLO_OUT_STRIDE + out_x * MLO_READ_UNIT;

    for(uint o = 0, gbl_out_off1 = gbl_out_off; o < MLO_N_LCL_OUT_MAPS;
        ++o, gbl_out_off1 += MLO_OUT_CHANNEL_STRIDE)
    {

        for(uint i = 0; i < MLO_READ_UNIT; ++i)
        {
			uint out_off = gbl_out_off1 + i
#if MLO_DIR_FORWARD == 0
				*MLO_FILTER_STRIDE0
#endif
			;
            out_ptr[out_off] = accum[o][i];
#if 0
				if (out_off == 14)
				{
					printf((__constant char *)"K:o0: %f %d %d %d\n",
					accum[o][i],
					MLO_READ_UNIT,
					get_global_id(0),
					i
					);
				}
#endif
#if MLO_DIR_FORWARD == 0
			for(uint s = 1; s < MLO_FILTER_STRIDE0; ++s)
			{
				out_ptr[out_off + s ] = 0;
#if 0
				if (out_off + s == 14)
				{
					printf((__constant char *)"K:o1: %d %d\n",
					get_global_id(0),
					s
					);
				}
#endif
			}
#endif
        }

#if MLO_DIR_FORWARD == 0
		for(uint j = 1; j < MLO_FILTER_STRIDE1; ++j)
		{
			uint out_off = gbl_out_off1 + j*MLO_OUT_STRIDE;
			for(uint s = 0; s < MLO_READ_UNIT* MLO_FILTER_STRIDE0; ++s)
			{
				out_ptr[out_off + s ] = 0;
#if 0
				if (out_off + s == 14)
				{
					printf((__constant char *)"K:o2: %d %d\n",
					get_global_id(0),
					j
					);
				}
#endif			
			}

		}
#endif
    }
}
