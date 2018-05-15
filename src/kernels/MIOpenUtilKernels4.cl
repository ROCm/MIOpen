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

inline uint iMod(uint v, uint u, uint d)
{
    uint r = v - mul24(u, d);
    return (r);
}

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

#ifdef NC_TRANS_NCHW_OPT
__kernel void transpose_NCHW2CNHW_opt(const global float* in, global float* out)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c        = c_p_blck / HW_RD;
    uint p_blck   = iMod(c_p_blck, c, HW_RD);

    uint in_off                 = c_p_blck * RD_BLCK + IN_OFF;
    uint out_off                = c * N * H * W + p_blck * RD_BLCK + OUT_OFF;
    const global READ_TYPE* cin = (const global READ_TYPE*)(in + in_off);
    global READ_TYPE* cout      = (global READ_TYPE*)(out + out_off);

    int b;
#ifdef _2D_WG
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

#ifdef NC_TRANS_CNHW_OPT
__kernel void transpose_CNHW2NCHW_opt(const global float* in, global float* out)
{
    // to reduce granularity loss
    uint c_p_blck = get_global_id(0);
    uint c        = c_p_blck / HW_RD;
    uint p_blck   = iMod(c_p_blck, c, HW_RD);

    uint in_off                 = c * N * H * W + p_blck * RD_BLCK + IN_OFF;
    uint out_off                = c_p_blck * RD_BLCK + OUT_OFF;
    const global READ_TYPE* cin = (const global READ_TYPE*)(in + in_off);
    global READ_TYPE* cout      = (global READ_TYPE*)(out + out_off);

    int b;
#ifdef _2D_WG
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

#ifdef NC_TRANS_NCHW
__kernel void transpose_NCHW2CNHW(const global float* in, global float* out)
{
    uint i = get_global_id(0);

    uint hw_i = i % HW_OUT;
    uint h_i  = hw_i / W_OUT;
    uint w_i  = hw_i % W_OUT;
    uint c_i  = i / HW_OUT;

    uint in_off             = c_i * HW_IN + h_i * H_STRIDE * W_IN + w_i * W_STRIDE + IN_OFF;
    uint out_off            = c_i * N * HW_OUT + hw_i + OUT_OFF;
    const global float* cin = (const global float*)(in + in_off);
    global float* cout      = (global float*)(out + out_off);

    uint n_i;
#ifdef _2D_WG
    n_i                = get_global_id(1);
    cout[HW_OUT * n_i] = cin[C * HW_IN * n_i];
#else
    for(n_i                = 0; n_i < N; n_i++)
        cout[HW_OUT * n_i] = cin[C * HW_IN * n_i];
#endif
}
#endif

#ifdef NC_TRANS_CNHW
__kernel void transpose_CNHW2NCHW(const global float* in, global float* out)
{
    uint i = get_global_id(0);

    uint hw_i = i % HW_OUT;
    uint h_i  = hw_i / W_OUT;
    uint w_i  = hw_i % W_OUT;
    uint c_i  = i / HW_OUT;

    uint in_off             = c_i * N * HW_OUT + hw_i + IN_OFF;
    uint out_off            = c_i * HW_IN + h_i * H_STRIDE * W_IN + w_i * W_STRIDE + OUT_OFF;
    const global float* cin = (const global float*)(in + in_off);
    global float* cout      = (global float*)(out + out_off);

    uint n_i;
#ifdef _2D_WG
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
