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
 */

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#ifndef MIOPEN_USE_INT8x4
#define MIOPEN_USE_INT8x4 0
#endif

#if MIOPEN_USE_INT8
typedef char data_t;
#elif MIOPEN_USE_INT8x4
typedef uint data_t;
#elif MIOPEN_USE_FP16
// As the half type degrades the performance, use short instead of half in
// transpose kernels, which have no match op. May change back to half when
// compile can deliver equal performance as short
typedef short data_t;
#elif MIOPEN_USE_FP32
typedef float data_t;
#endif

#include "math_ops.h"

#ifndef NC_TRANS_NCHW_OPT
#define NC_TRANS_NCHW_OPT 0
#endif

#ifndef NC_TRANS_CNHW_OPT
#define NC_TRANS_CNHW_OPT 0
#endif

#ifndef NC_TRANS_NCHW
#define NC_TRANS_NCHW 0
#endif

#ifndef NC_TRANS_CNHW
#define NC_TRANS_CNHW 0
#endif

#ifndef NC_TRANS_MN2NM
#define NC_TRANS_MN2NM 0
#endif

#ifndef IS_2D_WG
#define IS_2D_WG 0
#endif

// N - batch size
// C - # of maps
// H - map height
// W - map width

// RD_BLCK = ((H*W)%8==0) ? 8 : ((H*W)%4==0) ? 4 : ((H*W)%3==0)? 3 : ((H*W)%2==0)? 2 : 1;
// HW_RD = (H*W)/RD_BLCK
// MAP_RD = HW_RD*C

// lcl size0 = ((MAP_RD + 63)/64 < 8) ? ((MAP_RD + 63)/64)*64 : 512;
// local size = (lcl size0, 1, 1)
// global size = (MAP_RD, N, 1)

#if NC_TRANS_NCHW_OPT
__kernel void transpose_NCHW2CNHW_opt(const global data_t* in, global data_t* out)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c        = iDiv(c_p_blck, HW_RD);
    uint p_blck   = iMod(c_p_blck, c, HW_RD);

    uint in_off                 = c_p_blck * RD_BLCK + IN_OFF;
    uint out_off                = c * N * H * W + p_blck * RD_BLCK + OUT_OFF;
    const global READ_TYPE* cin = (const global READ_TYPE*)(in + in_off);
    global READ_TYPE* cout      = (global READ_TYPE*)(out + out_off);

    int b;
#if IS_2D_WG
    b               = get_global_id(1);
    cout[b * HW_RD] = cin[b * C * HW_RD];
#else
    for(b = 0; b < N; b++)
    {
        cout[b * HW_RD] = cin[b * C * HW_RD];
    }
#endif
}
#endif

#if NC_TRANS_CNHW_OPT
__kernel void transpose_CNHW2NCHW_opt(const global data_t* in, global data_t* out)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c        = iDiv(c_p_blck, HW_RD);
    uint p_blck   = iMod(c_p_blck, c, HW_RD);

    uint in_off                 = c * N * H * W + p_blck * RD_BLCK + IN_OFF;
    uint out_off                = c_p_blck * RD_BLCK + OUT_OFF;
    const global READ_TYPE* cin = (const global READ_TYPE*)(in + in_off);
    global READ_TYPE* cout      = (global READ_TYPE*)(out + out_off);

    int b;
#if IS_2D_WG
    b                   = get_global_id(1);
    cout[b * C * HW_RD] = cin[b * HW_RD];
#else
    for(b = 0; b < N; b++)
    {
        cout[b * C * HW_RD] = cin[b * HW_RD];
    }
#endif
}
#endif

#if NC_TRANS_NCHW
__kernel void transpose_NCHW2CNHW(const global data_t* in, global data_t* out)
{
    uint i = get_global_id(0);

    uint c_i  = iDiv(i, HW_OUT);
    uint hw_i = iMod(i, c_i, HW_OUT);
    uint h_i  = iDiv(hw_i, W_OUT);
    uint w_i  = iMod(hw_i, h_i, W_OUT);

    uint in_off              = c_i * HW_IN + h_i * H_STRIDE * W_IN + w_i * W_STRIDE + IN_OFF;
    uint out_off             = c_i * N * HW_OUT + hw_i + OUT_OFF;
    const global data_t* cin = (const global data_t*)(in + in_off);
    global data_t* cout      = (global data_t*)(out + out_off);

    uint n_i;
#if IS_2D_WG
    n_i                = get_global_id(1);
    cout[HW_OUT * n_i] = cin[C * HW_IN * n_i];
#else
    for(n_i                = 0; n_i < N; n_i++)
        cout[HW_OUT * n_i] = cin[C * HW_IN * n_i];
#endif
}
#endif

#if NC_TRANS_CNHW
__kernel void transpose_CNHW2NCHW(const global data_t* in, global data_t* out)
{
    uint i = get_global_id(0);

    uint c_i  = iDiv(i, HW_OUT);
    uint hw_i = iMod(i, c_i, HW_OUT);
    uint h_i  = iDiv(hw_i, W_OUT);
    uint w_i  = iMod(hw_i, h_i, W_OUT);

    uint in_off              = c_i * N * HW_OUT + hw_i + IN_OFF;
    uint out_off             = c_i * HW_IN + h_i * H_STRIDE * W_IN + w_i * W_STRIDE + OUT_OFF;
    const global data_t* cin = (const global data_t*)(in + in_off);
    global data_t* cout      = (global data_t*)(out + out_off);

    uint n_i;
#if IS_2D_WG
    n_i                   = get_global_id(1);
    cout[C * HW_IN * n_i] = cin[HW_OUT * n_i];
#else
    for(n_i = 0; n_i < N; n_i++)
    {
        cout[C * HW_IN * n_i] = cin[HW_OUT * n_i];
    }
#endif
}
#endif

#if NC_TRANS_MN2NM
__kernel void transpose_packed_MN2NM(const global data_t* in, global data_t* out)
{
    uint i = get_global_id(0);

    if(i < M * N)
    {
        uint m_i = iDiv(i, N);
        uint n_i = iMod(i, m_i, N);

        uint in_off  = m_i * N + n_i + IN_OFF;
        uint out_off = n_i * M + m_i + OUT_OFF;

        const global data_t* cin = (const global data_t*)(in + in_off);
        global data_t* cout      = (global data_t*)(out + out_off);

        *cout = *cin;
    }
}
#endif
