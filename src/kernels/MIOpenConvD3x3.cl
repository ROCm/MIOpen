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

#define UNUSED __attribute__((__unused__))

#define DBG_OUT_OF_RNGE 0

#ifndef MLO_COMPILER_AMD_OPENCL_HSAIL
#define MLO_COMPILER_AMD_OPENCL_HSAIL 0
#endif

#ifndef MLO_COMPILER_AMD_OPENCL_LC
#define MLO_COMPILER_AMD_OPENCL_LC 0
#endif

#ifndef MLO_OUTPUTS_ALIGNED
#define MLO_OUTPUTS_ALIGNED 0
#endif

#define MLO_HW_WAVE_ID_SETTING 1
// FIXME Conduct enabling from the host code.
extern __attribute__((const)) uint __hsail_get_dynwave_id(void);

#define MLO_IN_LCL_WIDTH (MLO_IN_STRIDE + 2 * MLO_FILTER_PAD0)

static inline int getWaveId()
{
    int wave_id = 0;

#if MLO_HW_WAVE_ID_SETTING && MLO_COMPILER_AMD_OPENCL_HSAIL == 1
    wave_id = __hsail_get_dynwave_id();
    wave_id &= MLO_N_WAVES_MASK;
#elif MLO_HW_WAVE_ID_SETTING && MLO_COMPILER_AMD_OPENCL_LC == 1 && MLO_GRP_SZ1 == 1 && \
    MLO_GRP_SZ2 == 1 && (MLO_GRP_SZ % (1 << MLO_LG2_WAVE_SZ)) == 0
    // (local_id/wavesize) has the same value in all workitems.
    // Make it scalar to enable scalarization optimizations.
    wave_id = __llvm_amdgcn_readfirstlane((uint)(get_local_id(0) >> MLO_LG2_WAVE_SZ));
// Alternate implementation:
//__asm__ ("v_readfirstlane_b32 %0, %1" : "=s" (wave_id) : "v" ((int)(get_local_id(0) >>
// MLO_LG2_WAVE_SZ)) );
#else
    wave_id = (get_local_id(0) >> MLO_LG2_WAVE_SZ);
#endif
    return (wave_id);
}

static inline int getWaveLocalId()
{
    int lcl_wave_id = get_local_id(0) - ((get_local_id(0) >> MLO_LG2_WAVE_SZ) << MLO_LG2_WAVE_SZ);
    return (lcl_wave_id);
}

static inline int getWaveLocalIds(int* lcl_wave_id1, int* lcl_wave_id0)
{
    int lcl_wave_id = getWaveLocalId();
    *lcl_wave_id1   = (lcl_wave_id >> MLO_LG2_WAVE_SZ0);
    *lcl_wave_id0   = lcl_wave_id - ((*lcl_wave_id1) << MLO_LG2_WAVE_SZ0);
    return (lcl_wave_id);
}

static inline int getLocalIds(int* lcl_id1, int* lcl_id0)
{
    int lcl_wave_id = getWaveLocalId();
    int wave_id     = getWaveId();
    int lcl_id      = (wave_id << MLO_LG2_WAVE_SZ) + lcl_wave_id; // get_local_id(0);
    *lcl_id1        = (lcl_id >> MLO_LG2ALU_EXTENT_X);
    *lcl_id0        = lcl_id - ((*lcl_id1) << MLO_LG2ALU_EXTENT_X);
    return (lcl_id);
}

/*********************************************************************************************************

**********************************************************************************************************/

__attribute__((reqd_work_group_size(MLO_GRP_SZ0, MLO_GRP_SZ1, MLO_GRP_SZ2))) __kernel void
MIOpenCvD3x3_WSR0(const __global _FLOAT* __restrict in_ptr,
                  const __global _FLOAT* __restrict wei_ptr,
#if MLO_CONV_BIAS
                  const __global _FLOAT* __restrict bias,
#endif
                  __global _FLOAT* __restrict out_ptr,
                  UNUSED _FLOAT alpha // LRU fusion
                  )
{
    // all other allocation should be before this one since we may exceed lcl mem range
    __local _FLOAT in_lcl[MLO_IN_LCL_WIDTH * (MLO_OUT_EXTENT1 + 2 * MLO_FILTER_PAD1)];

    __private _FLOAT pvt_accum[MLO_N_LCL_OUT_MAPS * MLO_OUT_TILE1 * MLO_OUT_TILE0];

    int grp_input_id   = get_group_id(0); // tile id inside the input map
    int grp_gbl_offset = (grp_input_id == 0)
                             ? 0
                             : (grp_input_id * MLO_OUT_EXTENT1 - MLO_FILTER_PAD1) * MLO_IN_STRIDE;
    int gpr_lcl_off = (grp_input_id != 0) ? MLO_FILTER_PAD0
                                          : MLO_IN_LCL_WIDTH * MLO_FILTER_PAD1 + MLO_FILTER_PAD0;

    int lcl_id1;
    int lcl_id0;
    int lcl_id = getLocalIds(&lcl_id1, &lcl_id0);
    grp_gbl_offset += lcl_id1 * MLO_IN_STRIDE;
    gpr_lcl_off += lcl_id1 * MLO_IN_LCL_WIDTH;

    int n_vert_reads = (grp_input_id == 0) ? (MLO_OUT_EXTENT1 + MLO_FILTER_PAD1)
                                           : (MLO_OUT_EXTENT1 + 2 * MLO_FILTER_PAD1);
    n_vert_reads = (grp_input_id == (get_num_groups(0) - 1))
                       ? (MLO_IN_HEIGHT - grp_input_id * MLO_OUT_EXTENT1 + MLO_FILTER_PAD1)
                       : n_vert_reads;

    // output
    int o_block = get_group_id(1);

    int weave_id = getWaveId();
    // assumption is MLO_N_OUTPUTS is multiple of MLO_N_LCL_OUT_MAPS*MLO_N_WAVES
    int o_base = o_block * MLO_N_LCL_OUT_MAPS * MLO_N_WAVES + weave_id * MLO_N_LCL_OUT_MAPS;

    __private int weights_base_offsets[MLO_N_LCL_OUT_MAPS];

    for(uint i = 0; i < MLO_N_LCL_OUT_MAPS; ++i)
    {
        weights_base_offsets[i] = (o_base + i) * MLO_WEI_BATCH_STRIDE;
    }

    // batch
    int b_block = get_group_id(2);
    // batch
    int b_base = b_block * MLO_N_LCL_BATCHS;

    grp_gbl_offset += b_base * MLO_IN_BATCH_STRIDE;

    // padding

    for(uint i = lcl_id; i < MLO_IN_LCL_WIDTH * (MLO_OUT_EXTENT1 + 2 * MLO_FILTER_PAD1);
        i += MLO_GRP_SZ)
    {
        in_lcl[i] = 0;
    }

    // local read and relative output adresses and offsets
    int lcl_wave_id1;
    int lcl_wave_id0;
    getWaveLocalIds(&lcl_wave_id1, &lcl_wave_id0);
    int lcl_off1     = lcl_wave_id1 * MLO_OUT_TILE1;
    int lcl_off0     = lcl_wave_id0 * MLO_OUT_TILE0;
    int lcl_read_off = lcl_off1 * MLO_IN_LCL_WIDTH + lcl_off0;

    for(uint i = 0; i < MLO_N_LCL_OUT_MAPS * MLO_OUT_TILE1 * MLO_OUT_TILE0; ++i)
    {
        pvt_accum[i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#if 0

	if (get_group_id(0) == 0 && lcl_id0 == 1)
	{
		printf("k:s: %d %d %d %d %d\n",
			getWaveId(),
			lcl_id,
			lcl_id1,
			grp_gbl_offset,
			gpr_lcl_off
		);
	}
#endif

    // outer loop over all input maps
    for(uint c = 0, wei_offset = 0; c < MLO_N_INPUTS;
        ++c, grp_gbl_offset += MLO_IN_CHANNEL_STRIDE, wei_offset += MLO_WEI_CHANNEL_STRIDE)
    {

        barrier(CLK_LOCAL_MEM_FENCE);

        // this kernel assumes with is multiple of MLO_READ_UNIT
        // number of inputs has been guarded above

        int gbl_scan_off = grp_gbl_offset;
        int lcl_scan_off = gpr_lcl_off;
        for(uint j = lcl_id1; j < n_vert_reads; j += MLO_ALU_EXTENT_Y,
                 gbl_scan_off += MLO_IN_STRIDE * MLO_ALU_EXTENT_Y,
                 lcl_scan_off += MLO_IN_LCL_WIDTH * MLO_ALU_EXTENT_Y)
        {
            for(int k = 0; k < MLO_READ_UNIT; ++k)
            {
                in_lcl[lcl_scan_off + lcl_id0 * MLO_READ_UNIT + k] =
                    in_ptr[gbl_scan_off + lcl_id0 * MLO_READ_UNIT + k];
            }
#if 0

			if (get_group_id(0) == 0 && j < 3 && lcl_id0 == 0)
			{
				printf("k:i: %d %d %d  %f %f %f\n",
					n_vert_reads,
					lcl_scan_off + lcl_id0*MLO_READ_UNIT,
					gbl_scan_off + lcl_id0*MLO_READ_UNIT,
					in_lcl[lcl_scan_off - 1], in_lcl[lcl_scan_off + lcl_id0*MLO_READ_UNIT], in_lcl[lcl_scan_off + lcl_id0*MLO_READ_UNIT + 1]
				);
			}
#endif
        }
        // finish reading
        barrier(CLK_LOCAL_MEM_FENCE);

        // read weights
        __private _FLOAT pvt_wei_stage[MLO_N_LCL_OUT_MAPS * MLO_FILTER_SIZE0 * MLO_FILTER_SIZE0];
        for(uint j = 0; j < MLO_N_LCL_OUT_MAPS; ++j)
        {
            for(uint i = 0; i < MLO_WEI_CHANNEL_STRIDE; ++i)
            {
                int wei_gbl_off = weights_base_offsets[j] + wei_offset;

#if MLO_OUTPUTS_ALIGNED == 0
                if(o_base + j >= MLO_N_OUTPUTS)
                {
                    pvt_wei_stage[j * MLO_WEI_CHANNEL_STRIDE + i] = 0;
                }
                else
#endif
                {
                    pvt_wei_stage[j * MLO_WEI_CHANNEL_STRIDE + i] = wei_ptr[wei_gbl_off + i];
                }
            }
        }

        // read data
        __private _FLOAT pvt_in_stage[(MLO_OUT_TILE1 + 2 * MLO_FILTER_PAD1) *
                                      (MLO_OUT_TILE0 + 2 * MLO_FILTER_PAD0)];

        int lcl_read_scan_off = lcl_read_off;
        for(uint j = 0; j < (MLO_OUT_TILE1 + 2 * MLO_FILTER_PAD1);
            ++j, lcl_read_scan_off += MLO_IN_LCL_WIDTH)
        {
            for(uint i = 0; i < (MLO_OUT_TILE0 + 2 * MLO_FILTER_PAD0); ++i)
            {
                pvt_in_stage[j * (MLO_OUT_TILE0 + 2 * MLO_FILTER_PAD0) + i] =
                    in_lcl[lcl_read_scan_off + i];
            }
        }

        // actual conv

        for(uint oc = 0; oc < MLO_N_LCL_OUT_MAPS; ++oc)
        {
            for(uint j = 0; j < MLO_OUT_TILE1; ++j)
            {
                for(uint i = 0; i < MLO_OUT_TILE0; ++i)
                {
                    for(uint k = 0; k < MLO_FILTER_SIZE1; ++k)
                    {
                        for(uint l = 0; l < MLO_FILTER_SIZE0; ++l)
                        {

                            pvt_accum[(oc * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i] +=
                                pvt_in_stage[(j + k) * (MLO_OUT_TILE0 + 2 * MLO_FILTER_PAD0) + i +
                                             l] *
                                pvt_wei_stage[oc * MLO_WEI_CHANNEL_STRIDE + k * MLO_FILTER_SIZE0 +
                                              l];
#if 0

							if (get_group_id(0) == 0 && lcl_id == 0 && j == 0 && i == 0)
							{
								printf("k:c: %d %d %d  %f %f %f\n",
									oc,
									k,
									l,
									pvt_accum[(oc * MLO_OUT_TILE1 + j) * MLO_OUT_TILE0 + i],
									pvt_in_stage[(j + k) * (MLO_OUT_TILE0 + 2 * MLO_FILTER_PAD0) + i + l],
									pvt_wei_stage[oc * MLO_WEI_CHANNEL_STRIDE + k * MLO_FILTER_SIZE0 + l]
								);
							}
#endif

                        } // for (int l = 0; l < MLO_FILTER_SIZE0; ++l)
                    }

                } // for (int i = 0; i < MLO_OUT_TILE0; ++i)

            } // for (int j = 0; j < MLO_OUT_TILE1; ++j)
        }
    }

    // send it out
    // TODO: batch loop

    int grp_x   = 0;
    int grp_y   = grp_input_id * MLO_OUT_EXTENT1;
    int out_off = b_base * MLO_OUT_BATCH_STRIDE + o_base * MLO_OUT_CHANNEL_STRIDE +
                  (grp_y + lcl_off1) * MLO_OUT_STRIDE + grp_x + lcl_off0;

    int out_off1 = out_off;
    for(uint oc = 0; oc < MLO_N_LCL_OUT_MAPS; ++oc, out_off1 += MLO_OUT_CHANNEL_STRIDE)
    {
#if MLO_OUTPUTS_ALIGNED == 0
        if(o_base + oc < MLO_N_OUTPUTS)
#endif
        {
            int out_off2 = out_off1;
            for(uint j = 0; j < MLO_OUT_TILE1; ++j, out_off2 += MLO_OUT_STRIDE)
            {
                // input height == output heiht
                if(grp_y + lcl_off1 + j < MLO_IN_HEIGHT)
                {
                    for(uint i = 0; i < MLO_OUT_TILE0 / MLO_READ_UNIT; ++i)
                    {
                        int gbl_out_off = out_off2 + i * MLO_READ_UNIT;
                        for(int k = 0; k < MLO_READ_UNIT; ++k)
                        {
                            out_ptr[gbl_out_off + k] =
                                pvt_accum[/*ib*MLO_N_LCL_OUT_MAPS * MLO_OUT_TILE_SZ + */ (
                                              oc * MLO_OUT_TILE1 + j) *
                                              MLO_OUT_TILE0 +
                                          i * MLO_READ_UNIT + k];
#if DBG_OUT_OF_RNGE
                            if(gbl_out_off + k > MLO_OUT_BATCH_STRIDE * MLO_BATCH_SZ)
                            {
                                printf("k:err:out-of-range\n");
                            }
#endif
                        }
                    }
                }
            }
        }
    }
}
