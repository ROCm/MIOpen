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
#if MLO_N_PREFETCHED > 1
	__local
#endif
	_FLOAT weights[MLO_N_LCL_OUT_MAPS][MLO_N_LCL_IN_MAPS*MLO_N_PREFETCHED]; 

    uint gbl_id0       = get_global_id(0);    
    uint batch_id      = gbl_id0 / MLO_MAP_SZ4; // batch
	uint pos           = gbl_id0 % MLO_MAP_SZ4;

    uint out_grp_block = get_group_id(1);      // block of outputs for the entire group
	uint out_id = out_grp_block * MLO_N_LCL_OUT_MAPS;


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
				 gbl_in_off += MLO_N_PREFETCHED * MLO_N_LCL_IN_MAPS * MLO_IN_CHANNEL_STRIDE,

// move weights offset
				 wei_off +=
				MLO_N_PREFETCHED * MLO_N_LCL_IN_MAPS *
#if MLO_DIR_FORWARD == 1
				MLO_WEI_CHANNEL_STRIDE   
#else
				MLO_WEI_BSTRIDE
#endif
	)
	{
// read weights (with prefetch)

#if MLO_N_PREFETCHED > 1
		for(uint o = 0, wei_off1 = wei_off; o < MLO_N_LCL_OUT_MAPS; ++o, wei_off1 += 
#if MLO_DIR_FORWARD == 1
                   MLO_WEI_BSTRIDE
#else
                   MLO_WEI_CHANNEL_STRIDE
#endif
		)
		{
			uint lcl_id0 = get_local_id(0);
			for(uint c = lcl_id0, wei_off2 = wei_off1 + lcl_id0 *
#if MLO_DIR_FORWARD == 1
				MLO_WEI_CHANNEL_STRIDE   
#else
				MLO_WEI_BSTRIDE
#endif

			; c < MLO_N_PREFETCHED * MLO_N_LCL_IN_MAPS;
			c += MLO_GRP_SZ0, wei_off2 += MLO_GRP_SZ0 *
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

#else
		for(uint o = 0, wei_off1 = wei_off; o < MLO_N_LCL_OUT_MAPS; ++o, wei_off1 += 
#if MLO_DIR_FORWARD == 1
                   MLO_WEI_BSTRIDE
#else
                   MLO_WEI_CHANNEL_STRIDE
#endif
		)
		{
			for(uint c = 0, wei_off2 = wei_off1; c < MLO_N_PREFETCHED * MLO_N_LCL_IN_MAPS; ++c, wei_off2 +=
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
#endif

// convolve with all prefetched waights
		for(int p = 0, gbl_in_off0 = gbl_in_off; p < MLO_N_PREFETCHED; ++p, gbl_in_off0 += MLO_N_LCL_IN_MAPS * MLO_IN_CHANNEL_STRIDE)
		{
// read data
			for(uint j = 0, gbl_in_off1 = gbl_in_off0; j < MLO_N_LCL_IN_MAPS; ++j, gbl_in_off1 += MLO_IN_CHANNEL_STRIDE)
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
						accum[o][i] += dat[c][i] * weights[o][p*MLO_N_LCL_IN_MAPS + c];
					}	
				}
			}
		}
	
		
	}



    uint gbl_out_off = batch_id * MLO_OUT_BATCH_STRIDE + pos * MLO_READ_UNIT + out_id * MLO_OUT_CHANNEL_STRIDE;
	for(uint o = 0, gbl_out_off1 = gbl_out_off; o < MLO_N_LCL_OUT_MAPS; ++o, gbl_out_off1 += MLO_OUT_CHANNEL_STRIDE)
	{
		for(uint i = 0; i < MLO_READ_UNIT; ++i)
		{
		 out_ptr[gbl_out_off1 + i] = accum[o][i];
		}
	}

}
