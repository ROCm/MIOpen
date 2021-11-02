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

// Trying to use float ATOMIC_ADD to increase total waves
// For example,  7x7 : 49
// Global 0:  H * W * N * (K/ 16 outplane per threads ) * (( Inputlanes/ 128 )   or 1)
// N * H * W =
//    [groupId / ((K/ 16 outplane per threads ) * (( Inputlanes/ 128 )   or 1))] * 64 + localId
// For example, H7*W7*N*C256*K64
// FIRST 4 waves
// Hit L2 for read
// try to Hit L2 for Atomic_ADD
// Assembly Shader can utlize LDS to Reductiion
// However Shader Compiler of OpenCL will compile any localId[1] to FLAT_BUFFER_LOAD not constant
// load

// WAVE 0  == N0_7x7 + N1_(7x2+1),  output  0-15  from K=64,  Inputplanes from 0-127
// WAVE 1  == N0_7x7 + N1_(7x2+1),  output  16-31 from K=64,  Inputplanes from 0-127
// WAVE 2  == N0_7x7 + N1_(7x2+1),  output  31-47 from K=64,  Inputplanes from 0-127
// WAVE 3  == N0_7x7 + N1_(7x2+1),  output  48-63 from K=64,  Inputplanes from 0-127
// 2nd 4 waves
// Hit L2 for read
// try to Hit L2 for Atomic_ADD

// WAVE 4  == N0_7x7 + N1_(7x2+1),  output  0-15  from K=64,  Inputplanes from 128-255
// WAVE 5  == N0_7x7 + N1_(7x2+1),  output  16-31 from K=64,  Inputplanes from 128-255
// WAVE 6  == N0_7x7 + N1_(7x2+1),  output  31-47 from K=64,  Inputplanes from 128-255
// WAVE 7  == N0_7x7 + N1_(7x2+1),  output  48-63 from K=64,  Inputplanes from 128-255

// STRIDE Mode: Global 0:  H_out * W_out * N * (K/ 16 outplane per threads ) * (( Inputlanes/ 128 )
// or 1)

// example
#if 0 // ndef MLopen_RUNNING
#define MLO_FILTER_STRIDE0 2
#define MLO_FILTER_STRIDE1 2
#define MLO_N_LCL_IN_MAPS_ONCE 8

#define H 28
#define W 28
#define C 192
#define K 64

#define MLO_IN_HEIGHT H
#define MLO_IN_WIDTH W
#define MLO_N_INPUTS C

//128 or MLO_N_INPUTS
#define MLO_N_LCL_IN_MAPS 192

#define MLO_N_OUTPUTS K

#define H_out 28
#define W_out 28
#define MLO_N_LCL_OUT_MAPS 16

#define MLO_N_IN_GROUPS ((MLO_N_INPUTS + MLO_N_LCL_IN_MAPS - 1) / MLO_N_LCL_IN_MAPS)
#define MLO_CLOOP0 (MLO_N_LCL_IN_MAPS / MLO_N_LCL_IN_MAPS_ONCE)
#define MLO_CLOOP2 \
    ((MLO_N_INPUTS - MLO_N_LCL_IN_MAPS * (MLO_N_IN_GROUPS - 1)) / MLO_N_LCL_IN_MAPS_ONCE)
#define MLO_CHEAT_SHADER_COMPILER 1

#endif

#define MLO_IN_CHANNEL_STRIDE (H * W)
#define MLO_IN_BATCH_STRIDE (H * W * C)

#define MLO_WEI_BSTRIDE (1 * 1 * C * K)
#define MLO_WEI_CHANNEL_STRIDE (1 * 1 * C)

#define MLO_OUT_BATCH_STRIDE (H_out * W_out * K)
#define MLO_OUT_CHANNEL_STRIDE (H_out * W_out)

#define FIXED_WORKGROUP_SIZE 64

#define MLO_N_OUT_GROUPS (MLO_N_OUTPUTS / MLO_N_LCL_OUT_MAPS)

#define MLO_GRP_SZ0 64
#define MLO_GRP_SZ1 1
#define MLO_GRP_SZ2 1

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)
#define TWO 2
#define FOUR 4
#define EIGHT 8

#if MIOPEN_USE_FP16 == 1
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _FLOAT half
#define _UNION_FLOAT_T half2
#define INIT(A) ((half2)(A[0], A[1]))
#ifndef HALF_MAX
#define MAX_VAL 65504 /* max value */
#else
#define MAX_VAL HALF_MAX
#endif
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#define _UNION_FLOAT_T float
#define INIT(A) (A[0])
#ifndef FLT_MAX
#define MAX_VAL 3.402823466e+38F /* max value */
#else
#define MAX_VAL FLT_MAX
#endif
#endif

#define _FLOAT2 PPCAT(_FLOAT, TWO)
#define _FLOAT4 PPCAT(_FLOAT, FOUR)
#define _FLOAT8 PPCAT(_FLOAT, EIGHT)

#define MLO_CONV_BIAS 0
#define UNUSED __attribute((__unused__))

typedef union
{
    unsigned int intVal;
    _UNION_FLOAT_T floatVal;
} starVal;

inline void AtomicAdd(volatile __global _FLOAT* source, const _FLOAT operand)
{
    starVal newVal, prevVal;

    prevVal.floatVal = INIT(source);
    while(true)
    {
#if MIOPEN_USE_FP16 == 1
        newVal.floatVal = (_FLOAT2)(prevVal.floatVal.x + operand, source[1]);
#endif
#if MIOPEN_USE_FP32 == 1
        newVal.floatVal = prevVal.floatVal + operand;
#endif
        newVal.intVal =
            atomic_cmpxchg((volatile __global unsigned int*)source, prevVal.intVal, newVal.intVal);

        // equal to pass
        if(newVal.intVal == prevVal.intVal)
            break;

        prevVal.intVal = newVal.intVal;
    }
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

    uint grp_id0       = get_group_id(0);
    uint out_grp_block = grp_id0 % MLO_N_OUT_GROUPS;
    uint in_grp_block  = (uint)(grp_id0 / MLO_N_OUT_GROUPS) % MLO_N_IN_GROUPS;
    uint grp_id0_faked = (uint)(grp_id0 / MLO_N_OUT_GROUPS) / MLO_N_IN_GROUPS;

    uint local_id0 = get_local_id(0);
#if MLO_CHEAT_SHADER_COMPILER == 1
    uint grp_id2 = get_group_id(2);
#endif

    uint pos      = (grp_id0_faked * FIXED_WORKGROUP_SIZE + local_id0) % MLO_OUT_CHANNEL_STRIDE;
    uint batch_id = (grp_id0_faked * FIXED_WORKGROUP_SIZE + local_id0) / MLO_OUT_CHANNEL_STRIDE;

    if(batch_id >= BATCHSIZE)
        return;

    uint out_id = out_grp_block * MLO_N_LCL_OUT_MAPS;

    short out_pos_x = pos % W_out;
    short out_pos_y = pos / W_out;

    uint in_pos = out_pos_x * MLO_FILTER_STRIDE0 + out_pos_y * MLO_FILTER_STRIDE1 * W;

    uint gbl_in_off = batch_id * MLO_IN_BATCH_STRIDE +
                      in_grp_block * MLO_N_LCL_IN_MAPS * MLO_IN_CHANNEL_STRIDE + in_pos;

    uint wei_off = out_id * MLO_WEI_CHANNEL_STRIDE + in_grp_block * MLO_N_LCL_IN_MAPS;

    _FLOAT accum[MLO_N_LCL_OUT_MAPS];
    _FLOAT weights[MLO_N_LCL_IN_MAPS_ONCE];
    _FLOAT dat[MLO_N_LCL_IN_MAPS_ONCE];
    _FLOAT dat2[MLO_N_LCL_IN_MAPS_ONCE];

//

// ATOMIC is needed if INPUTS in many waves
#if(MLO_N_LCL_IN_MAPS != MLO_N_INPUTS)

    if(in_grp_block == 0)
    {
        uint gbl_out_off = batch_id * MLO_OUT_BATCH_STRIDE + out_id * MLO_OUT_CHANNEL_STRIDE + pos;
        __global _FLOAT* q = out_ptr + gbl_out_off;

        for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
        {
            *q = (_FLOAT)0;
            q += MLO_OUT_CHANNEL_STRIDE;
        }
    }
#endif

    for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
    {
        accum[o] = (_FLOAT)0;
    }

#if MLO_N_INPUTS == ((MLO_N_INPUTS / MLO_N_LCL_IN_MAPS) * MLO_N_LCL_IN_MAPS)
    // if(1)
    int loops = MLO_CLOOP0;

#if MLO_CHEAT_SHADER_COMPILER == 1
    // cheat shader compiler to disable loop unroll.  it will have better SQC performance
    if(grp_id2 == 0x1F)
    {
        loops = 377; // strange not to unroll loop
    }
#endif
#else
    int loops = MLO_CLOOP0;

    if(in_grp_block == (MLO_N_IN_GROUPS - 1))
    {
        loops = MLO_CLOOP2;
    }

#if MLO_CHEAT_SHADER_COMPILER == 1
    // cheat shader compiler to disable loop unroll.  it will have better SQC performance
    if(grp_id2 == 0x1F)
    {
        loops = 377; // strange not to unroll loop
    }
#endif

#endif
    {
        __global const _FLOAT* p = in_ptr + gbl_in_off;
        __constant _FLOAT* w     = wei_ptr + wei_off;

        // read data
        for(uint j = 0; j < MLO_N_LCL_IN_MAPS_ONCE; ++j)
        {

            dat[j] = *p;
            p += MLO_IN_CHANNEL_STRIDE;
        }

        for(uint ci = 0; ci < (loops - 2); ci += 2)
        {
            // read data
            for(uint j = 0; j < MLO_N_LCL_IN_MAPS_ONCE; ++j)
            {
                dat2[j] = *p;
                p += MLO_IN_CHANNEL_STRIDE;
            }

            // convolve
            __constant _FLOAT* w1 = w;
            for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
            {

                __constant _FLOAT* w2 = w1;

                for(uint j = 0; j < MLO_N_LCL_IN_MAPS_ONCE; ++j)
                {
                    weights[j] = *w2;
                    w2++;
                }
                w1 += MLO_WEI_CHANNEL_STRIDE;

                for(uint c = 0; c < MLO_N_LCL_IN_MAPS_ONCE; ++c)
                {
                    accum[o] += dat[c] * weights[c];
                }
            }

            // move weights offset
            w += MLO_N_LCL_IN_MAPS_ONCE;

            // convolve
            w1 = w;
            for(uint j = 0; j < MLO_N_LCL_IN_MAPS_ONCE; ++j)
            {
                dat[j] = *p;
                p += MLO_IN_CHANNEL_STRIDE;
            }

            for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
            {

                __constant _FLOAT* w2 = w1;

                for(uint j = 0; j < MLO_N_LCL_IN_MAPS_ONCE; ++j)
                {
                    weights[j] = *w2;
                    w2++;
                }
                w1 += MLO_WEI_CHANNEL_STRIDE;

                for(uint c = 0; c < MLO_N_LCL_IN_MAPS_ONCE; ++c)
                {
                    accum[o] += dat2[c] * weights[c];
                }
            }

            // move weights offset
            w += MLO_N_LCL_IN_MAPS_ONCE;
        }

        //
        // last 2 iterations
        { // read data
            for(uint j = 0; j < MLO_N_LCL_IN_MAPS_ONCE; ++j)
            {
                dat2[j] = *p;
                p += MLO_IN_CHANNEL_STRIDE;
            }

            // convolve
            __constant _FLOAT* w1 = w;
            for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
            {

                __constant _FLOAT* w2 = w1;

                for(uint j = 0; j < MLO_N_LCL_IN_MAPS_ONCE; ++j)
                {
                    weights[j] = *w2;
                    w2++;
                }
                w1 += MLO_WEI_CHANNEL_STRIDE;

                for(uint c = 0; c < MLO_N_LCL_IN_MAPS_ONCE; ++c)
                {
                    accum[o] += dat[c] * weights[c];
                }
            }

            // move weights offset
            w += MLO_N_LCL_IN_MAPS_ONCE;

            // convolve
            w1 = w;

            for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
            {

                __constant _FLOAT* w2 = w1;

                for(uint j = 0; j < MLO_N_LCL_IN_MAPS_ONCE; ++j)
                {
                    weights[j] = *w2;
                    w2++;
                }
                w1 += MLO_WEI_CHANNEL_STRIDE;

                for(uint c = 0; c < MLO_N_LCL_IN_MAPS_ONCE; ++c)
                {
                    accum[o] += dat2[c] * weights[c];
                }
            }

            // move weights offset
            w += MLO_N_LCL_IN_MAPS_ONCE;
        }
    }

    uint gbl_out_off   = batch_id * MLO_OUT_BATCH_STRIDE + out_id * MLO_OUT_CHANNEL_STRIDE + pos;
    __global _FLOAT* q = out_ptr + gbl_out_off;

    for(uint o = 0; o < MLO_N_LCL_OUT_MAPS; ++o)
    {
#if(MLO_N_LCL_IN_MAPS == MLO_N_INPUTS)
        *q = accum[o];
        q += MLO_OUT_CHANNEL_STRIDE;
#else
        AtomicAdd(q, accum[o]);
        q += MLO_OUT_CHANNEL_STRIDE;
#endif
    }
}
