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

#ifndef MIOPEN_USE_BFP16
#define MIOPEN_USE_BFP16 0
#endif

#ifndef MIOPEN_USE_INT8
#define MIOPEN_USE_INT8 0
#endif

#ifndef MIOPEN_USE_INT8x4
#define MIOPEN_USE_INT8x4 0
#endif

#ifndef MIOPEN_USE_INT32
#define MIOPEN_USE_INT32 0
#endif

#if MIOPEN_USE_INT8
typedef char data_t;
#elif MIOPEN_USE_INT8x4
typedef uint data_t;
#elif MIOPEN_USE_INT32
typedef int data_t;
#elif(MIOPEN_USE_FP16 || MIOPEN_USE_BFP16)
// As the half type degrades the performance, use short instead of half in
// transpose kernels, which have no match op. May change back to half when
// compile can deliver equal performance as short
typedef short data_t;
#elif MIOPEN_USE_FP32
typedef float data_t;
#endif

#include "math_ops.h"

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

__kernel void transpose_NCHW2CNHW_V1_1D_WG_float(const global data_t* in,
                                                 global data_t* out,
                                                 const int in_off,
                                                 const int out_off,
                                                 const int rd_blck,
                                                 const int hw_rd,
                                                 const int N,
                                                 const int C,
                                                 const int H,
                                                 const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset          = c_p_blck * rd_blck + in_off;
    uint out_offset         = c_i * N * H * W + p_blck * rd_blck + out_off;
    const global float* cin = (const global float*)(in + in_offset);
    global float* cout      = (global float*)(out + out_offset);

    for(uint b = 0; b < N; b++)
    {
        cout[b * hw_rd] = cin[b * C * hw_rd];
    }
}

__kernel void transpose_NCHW2CNHW_V1_1D_WG_float2(const global data_t* in,
                                                  global data_t* out,
                                                  const int in_off,
                                                  const int out_off,
                                                  const int rd_blck,
                                                  const int hw_rd,
                                                  const int N,
                                                  const int C,
                                                  const int H,
                                                  const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset           = c_p_blck * rd_blck + in_off;
    uint out_offset          = c_i * N * H * W + p_blck * rd_blck + out_off;
    const global float2* cin = (const global float2*)(in + in_offset);
    global float2* cout      = (global float2*)(out + out_offset);

    for(uint b = 0; b < N; b++)
    {
        cout[b * hw_rd] = cin[b * C * hw_rd];
    }
}

__kernel void transpose_NCHW2CNHW_V1_1D_WG_float4(const global data_t* in,
                                                  global data_t* out,
                                                  const int in_off,
                                                  const int out_off,
                                                  const int rd_blck,
                                                  const int hw_rd,
                                                  const int N,
                                                  const int C,
                                                  const int H,
                                                  const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset           = c_p_blck * rd_blck + in_off;
    uint out_offset          = c_i * N * H * W + p_blck * rd_blck + out_off;
    const global float4* cin = (const global float4*)(in + in_offset);
    global float4* cout      = (global float4*)(out + out_offset);

    for(uint b = 0; b < N; b++)
    {
        cout[b * hw_rd] = cin[b * C * hw_rd];
    }
}

__kernel void transpose_NCHW2CNHW_V1_2D_WG_float(const global data_t* in,
                                                 global data_t* out,
                                                 const int in_off,
                                                 const int out_off,
                                                 const int rd_blck,
                                                 const int hw_rd,
                                                 const int N,
                                                 const int C,
                                                 const int H,
                                                 const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset          = c_p_blck * rd_blck + in_off;
    uint out_offset         = c_i * N * H * W + p_blck * rd_blck + out_off;
    const global float* cin = (const global float*)(in + in_offset);
    global float* cout      = (global float*)(out + out_offset);

    uint b          = get_global_id(1);
    cout[b * hw_rd] = cin[b * C * hw_rd];
}

__kernel void transpose_NCHW2CNHW_V1_2D_WG_float2(const global data_t* in,
                                                  global data_t* out,
                                                  const int in_off,
                                                  const int out_off,
                                                  const int rd_blck,
                                                  const int hw_rd,
                                                  const int N,
                                                  const int C,
                                                  const int H,
                                                  const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset           = c_p_blck * rd_blck + in_off;
    uint out_offset          = c_i * N * H * W + p_blck * rd_blck + out_off;
    const global float2* cin = (const global float2*)(in + in_offset);
    global float2* cout      = (global float2*)(out + out_offset);

    uint b          = get_global_id(1);
    cout[b * hw_rd] = cin[b * C * hw_rd];
}

__kernel void transpose_NCHW2CNHW_V1_2D_WG_float4(const global data_t* in,
                                                  global data_t* out,
                                                  const int in_off,
                                                  const int out_off,
                                                  const int rd_blck,
                                                  const int hw_rd,
                                                  const int N,
                                                  const int C,
                                                  const int H,
                                                  const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset           = c_p_blck * rd_blck + in_off;
    uint out_offset          = c_i * N * H * W + p_blck * rd_blck + out_off;
    const global float4* cin = (const global float4*)(in + in_offset);
    global float4* cout      = (global float4*)(out + out_offset);

    uint b          = get_global_id(1);
    cout[b * hw_rd] = cin[b * C * hw_rd];
}

__kernel void transpose_NCHW2CNHW_V2_2D_WG(const global data_t* in,
                                           global data_t* out,
                                           const int in_off,
                                           const int out_off,
                                           const int w_in,
                                           const int w_out,
                                           const int N,
                                           const int C,
                                           const int h_stride,
                                           const int w_stride,
                                           const int hw_in,
                                           const int hw_out)
{
    uint hw_i = get_global_id(0);
    uint c_i  = get_global_id(2);

    uint h_i = iDiv(hw_i, w_out);
    uint w_i = iMod(hw_i, h_i, w_out);

    uint in_offset           = c_i * hw_in + h_i * h_stride * w_in + w_i * w_stride + in_off;
    uint out_offset          = c_i * N * hw_out + hw_i + out_off;
    const global data_t* cin = (const global data_t*)(in + in_offset);
    global data_t* cout      = (global data_t*)(out + out_offset);

    for(uint n_i = 0; n_i < N; n_i++)
        cout[hw_out * n_i] = cin[C * hw_in * n_i];
}

__kernel void transpose_NCHW2CNHW_V2_3D_WG(const global data_t* in,
                                           global data_t* out,
                                           const int in_off,
                                           const int out_off,
                                           const int w_in,
                                           const int w_out,
                                           const int N,
                                           const int C,
                                           const int h_stride,
                                           const int w_stride,
                                           const int hw_in,
                                           const int hw_out)
{
    uint hw_i = get_global_id(0);
    uint c_i  = get_global_id(2);

    // uint c_i  = iDiv(i, hw_out);
    // uint hw_i = iMod(i, c_i, hw_out);
    uint h_i = iDiv(hw_i, w_out);
    uint w_i = iMod(hw_i, h_i, w_out);

    uint in_offset           = c_i * hw_in + h_i * h_stride * w_in + w_i * w_stride + in_off;
    uint out_offset          = c_i * N * hw_out + hw_i + out_off;
    const global data_t* cin = (const global data_t*)(in + in_offset);
    global data_t* cout      = (global data_t*)(out + out_offset);

    uint n_i           = get_global_id(1);
    cout[hw_out * n_i] = cin[C * hw_in * n_i];
}

__kernel void transpose_CNHW2NCHW_V1_1D_WG_float(const global data_t* in,
                                                 global data_t* out,
                                                 const int in_off,
                                                 const int out_off,
                                                 const int rd_blck,
                                                 const int hw_rd,
                                                 const int N,
                                                 const int C,
                                                 const int H,
                                                 const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset          = c_i * N * H * W + p_blck * rd_blck + in_off;
    uint out_offset         = c_p_blck * rd_blck + out_off;
    const global float* cin = (const global float*)(in + in_offset);
    global float* cout      = (global float*)(out + out_offset);

    for(uint b = 0; b < N; b++)
    {
        cout[b * C * hw_rd] = cin[b * hw_rd];
    }
}

__kernel void transpose_CNHW2NCHW_V1_1D_WG_float2(const global data_t* in,
                                                  global data_t* out,
                                                  const int in_off,
                                                  const int out_off,
                                                  const int rd_blck,
                                                  const int hw_rd,
                                                  const int N,
                                                  const int C,
                                                  const int H,
                                                  const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset           = c_i * N * H * W + p_blck * rd_blck + in_off;
    uint out_offset          = c_p_blck * rd_blck + out_off;
    const global float2* cin = (const global float2*)(in + in_offset);
    global float2* cout      = (global float2*)(out + out_offset);

    for(uint b = 0; b < N; b++)
    {
        cout[b * C * hw_rd] = cin[b * hw_rd];
    }
}

__kernel void transpose_CNHW2NCHW_V1_1D_WG_float4(const global data_t* in,
                                                  global data_t* out,
                                                  const int in_off,
                                                  const int out_off,
                                                  const int rd_blck,
                                                  const int hw_rd,
                                                  const int N,
                                                  const int C,
                                                  const int H,
                                                  const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset           = c_i * N * H * W + p_blck * rd_blck + in_off;
    uint out_offset          = c_p_blck * rd_blck + out_off;
    const global float4* cin = (const global float4*)(in + in_offset);
    global float4* cout      = (global float4*)(out + out_offset);

    for(uint b = 0; b < N; b++)
    {
        cout[b * C * hw_rd] = cin[b * hw_rd];
    }
}

__kernel void transpose_CNHW2NCHW_V1_2D_WG_float(const global data_t* in,
                                                 global data_t* out,
                                                 const int in_off,
                                                 const int out_off,
                                                 const int rd_blck,
                                                 const int hw_rd,
                                                 const int N,
                                                 const int C,
                                                 const int H,
                                                 const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset          = c_i * N * H * W + p_blck * rd_blck + in_off;
    uint out_offset         = c_p_blck * rd_blck + out_off;
    const global float* cin = (const global float*)(in + in_offset);
    global float* cout      = (global float*)(out + out_offset);

    uint b              = get_global_id(1);
    cout[b * C * hw_rd] = cin[b * hw_rd];
}

__kernel void transpose_CNHW2NCHW_V1_2D_WG_float2(const global data_t* in,
                                                  global data_t* out,
                                                  const int in_off,
                                                  const int out_off,
                                                  const int rd_blck,
                                                  const int hw_rd,
                                                  const int N,
                                                  const int C,
                                                  const int H,
                                                  const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset           = c_i * N * H * W + p_blck * rd_blck + in_off;
    uint out_offset          = c_p_blck * rd_blck + out_off;
    const global float2* cin = (const global float2*)(in + in_offset);
    global float2* cout      = (global float2*)(out + out_offset);

    uint b              = get_global_id(1);
    cout[b * C * hw_rd] = cin[b * hw_rd];
}

__kernel void transpose_CNHW2NCHW_V1_2D_WG_float4(const global data_t* in,
                                                  global data_t* out,
                                                  const int in_off,
                                                  const int out_off,
                                                  const int rd_blck,
                                                  const int hw_rd,
                                                  const int N,
                                                  const int C,
                                                  const int H,
                                                  const int W)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c_i      = iDiv(c_p_blck, hw_rd);
    uint p_blck   = iMod(c_p_blck, c_i, hw_rd);

    uint in_offset           = c_i * N * H * W + p_blck * rd_blck + in_off;
    uint out_offset          = c_p_blck * rd_blck + out_off;
    const global float4* cin = (const global float4*)(in + in_offset);
    global float4* cout      = (global float4*)(out + out_offset);

    uint b              = get_global_id(1);
    cout[b * C * hw_rd] = cin[b * hw_rd];
}

__kernel void transpose_CNHW2NCHW_V2_2D_WG(const global data_t* in,
                                           global data_t* out,
                                           const int in_off,
                                           const int out_off,
                                           const int w_in,
                                           const int w_out,
                                           const int N,
                                           const int C,
                                           const int h_stride,
                                           const int w_stride,
                                           const int hw_in,
                                           const int hw_out)
{
    uint hw_i = get_global_id(0);
    uint c_i  = get_global_id(2);

    uint h_i = iDiv(hw_i, w_out);
    uint w_i = iMod(hw_i, h_i, w_out);

    uint in_offset           = c_i * N * hw_out + hw_i + in_off;
    uint out_offset          = c_i * hw_in + h_i * h_stride * w_in + w_i * w_stride + out_off;
    const global data_t* cin = (const global data_t*)(in + in_offset);
    global data_t* cout      = (global data_t*)(out + out_offset);

    for(uint n_i = 0; n_i < N; n_i++)
    {
        cout[C * hw_in * n_i] = cin[hw_out * n_i];
    }
}

__kernel void transpose_CNHW2NCHW_V2_3D_WG(const global data_t* in,
                                           global data_t* out,
                                           const int in_off,
                                           const int out_off,
                                           const int w_in,
                                           const int w_out,
                                           const int N,
                                           const int C,
                                           const int h_stride,
                                           const int w_stride,
                                           const int hw_in,
                                           const int hw_out)
{
    uint hw_i = get_global_id(0);
    uint c_i  = get_global_id(2);

    uint h_i = iDiv(hw_i, w_out);
    uint w_i = iMod(hw_i, h_i, w_out);

    uint in_offset           = c_i * N * hw_out + hw_i + in_off;
    uint out_offset          = c_i * hw_in + h_i * h_stride * w_in + w_i * w_stride + out_off;
    const global data_t* cin = (const global data_t*)(in + in_offset);
    global data_t* cout      = (global data_t*)(out + out_offset);

    uint n_i              = get_global_id(1);
    cout[C * hw_in * n_i] = cin[hw_out * n_i];
}

__kernel void transpose_packed_MN2NM(const global data_t* in,
                                     global data_t* out,
                                     const int N,
                                     const int M,
                                     const int in_off,
                                     const int out_off)
{
    uint i = get_global_id(0);

    if(i < M * N)
    {
        uint m_i = iDiv(i, N);
        uint n_i = iMod(i, m_i, N);

        uint in_offset  = m_i * N + n_i + in_off;
        uint out_offset = n_i * M + m_i + out_off;

        const global data_t* cin = (const global data_t*)(in + in_offset);
        global data_t* cout      = (global data_t*)(out + out_offset);

        *cout = *cin;
    }
}
