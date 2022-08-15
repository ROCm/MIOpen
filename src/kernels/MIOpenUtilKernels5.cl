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

#ifndef USE_ALPHA
#define USE_ALPHA 0
#endif
#ifndef USE_BETA
#define USE_BETA 0
#endif

__attribute__((always_inline)) uint iDiv(uint v, uint d)
{
    uint r = v / d;
    return (r);
}

__attribute__((always_inline)) uint iMod(uint v, uint u, uint d)
{
    uint r = v - mul24(u, d);
    return (r);
}

#define UNUSED __attribute__((__unused__))

static inline uint get_in_off(const uint p_blck, const uint n, const uint c)
{
#if TRANS

    const uint in_off =
#if FORWARD
        p_blck * RD_BLCK + c * HW + n * CHW * VEC_SIZE
#else
        p_blck * WR_BLCK + n * HW * VEC_SIZE + c * NHW_OUT
#endif
        ;

#else

    const uint in_off =
#if FORWARD
        p_blck * RD_BLCK + c * HW * VEC_SIZE + n * CHW
#else
        p_blck * WR_BLCK + c * HW * VEC_SIZE + n * CHW_OUT
#endif
        ;

#endif // end of #if TRANS

    return in_off;
}

static inline uint get_out_off(const uint p_blck, const uint n, const uint c)
{
#if TRANS

    const uint out_off =
#if FORWARD
        p_blck * WR_BLCK + n * HW * VEC_SIZE + c * NHW_OUT
#else
        p_blck * RD_BLCK + c * HW + n * CHW * VEC_SIZE
#endif
        ;

#else

    const uint out_off =
#if FORWARD
        p_blck * WR_BLCK + c * HW * VEC_SIZE + n * CHW_OUT
#else
        p_blck * RD_BLCK + c * HW * VEC_SIZE + n * CHW
#endif
        ;

#endif // end of #if TRANS

    return out_off;
}

static inline void load_data(const uint in_off,
#if !(FORWARD && TRANS)
                             UNUSED
#endif
                             const uint n,
#if !(FORWARD && !TRANS)
                             UNUSED
#endif
                             const uint c,
                             const global DATA_TYPE* in,
                             __private DATA_TYPE* in_buf)
{
#if FORWARD
#pragma unroll
    for(int v = 0; v < VEC_SIZE; v++)
    {
#if TRANS
        *((READ_TYPE*)(in_buf + RD_BLCK * v)) =
            ((n * VEC_SIZE + v) < N) ? *((const global READ_TYPE*)(in + in_off + CHW * v)) : 0;
#else
        *((READ_TYPE*)(in_buf + RD_BLCK * v)) =
            ((c * VEC_SIZE + v) < C) ? *((const global READ_TYPE*)(in + in_off + HW * v)) : 0;
#endif
    }
#else

    *((WRITE_TYPE*)in_buf) = *((const global WRITE_TYPE*)(in + in_off));

#endif
}

static inline void local_trans(__private DATA_TYPE* in_buf, __private DATA_TYPE* out_buf)
{
    for(int i = 0; i < RD_BLCK; i++)
    {
#pragma unroll
        for(int v = 0; v < VEC_SIZE; v++)
        {
#if FORWARD
            out_buf[i * VEC_SIZE + v] = in_buf[v * RD_BLCK + i];
#else
            out_buf[v * RD_BLCK + i] = in_buf[i * VEC_SIZE + v];
#endif
        }
    }
}

static inline void write_data(const uint out_off,
#if !(!FORWARD && TRANS)
                              UNUSED
#endif
                              const uint n,
#if !(!FORWARD && !TRANS)
                              UNUSED
#endif
                              const uint c,
                              global DATA_TYPE* out,
                              const __private DATA_TYPE* out_buf)
{

#if FORWARD

    *((global WRITE_TYPE*)(out + out_off)) = *((WRITE_TYPE*)out_buf);

#else

#pragma unroll
    for(int v = 0; v < VEC_SIZE; v++)
    {
#if TRANS
        if((n * VEC_SIZE + v) < N)
            *((global READ_TYPE*)(out + out_off + CHW * v)) =
                *((READ_TYPE*)(out_buf + RD_BLCK * v));
#else
        if((c * VEC_SIZE + v) < C)
            *((global READ_TYPE*)(out + out_off + HW * v)) = *((READ_TYPE*)(out_buf + RD_BLCK * v));
#endif
    }

#endif
}

static inline void global_trans(const uint in_off,
                                const uint out_off,
                                const uint p_blck,
#if !TRANS
                                UNUSED
#endif
                                const uint n,
#if TRANS
                                UNUSED
#endif
                                const uint c,
                                const global DATA_TYPE* in,
                                global DATA_TYPE* out)
{
    int HW_tail = iMod(HW, p_blck, RD_BLCK);

    for(int i = 0; i < HW_tail; i++)
    {
#pragma unroll
        for(int v = 0; v < VEC_SIZE; v++)
        {
#if FORWARD

#if TRANS
            out[out_off + i * VEC_SIZE + v] =
                ((n * VEC_SIZE + v) < N) ? in[in_off + CHW * v + i] : 0;
#else
            out[out_off + i * VEC_SIZE + v] =
                ((c * VEC_SIZE + v) < C) ? in[in_off + HW * v + i] : 0;
#endif

#else

#if TRANS
            if((n * VEC_SIZE + v) < N)
                out[out_off + CHW * v + i] = in[in_off + i * VEC_SIZE + v];
#else
            if((c * VEC_SIZE + v) < C)
                out[out_off + HW * v + i] = in[in_off + i * VEC_SIZE + v];
#endif

#endif
        }
    }
}

__kernel void transpose_NCHW2Vec(const global DATA_TYPE* in,
                                 global DATA_TYPE* out,
#if !USE_ALPHA
                                 UNUSED
#endif
                                 const float alpha,
#if !USE_BETA
                                 UNUSED
#endif
                                 const float beta)
{
    // to reduce granularity loss
    const uint c_p_blck = get_global_id(0);
    const uint c        = iDiv(c_p_blck, HW_RD);
    const uint p_blck   = iMod(c_p_blck, c, HW_RD);

    __private DATA_TYPE in_buf[RD_BLCK * VEC_SIZE];
    __private DATA_TYPE out_buf[RD_BLCK * VEC_SIZE];

#if IS_2D_WG
    const uint n = get_global_id(1);
#else
    for(uint n = 0; n < GD_1; n++)
#endif
    {
        uint in_off = get_in_off(p_blck, n, c);

        uint out_off = get_out_off(p_blck, n, c);

#if IS_HW_ODD
        if(p_blck < HW_RD - 1)
#endif
        {
            load_data(in_off, n, c, in, in_buf);

            local_trans(in_buf, out_buf);

            write_data(out_off, n, c, out, out_buf);
        }
#if IS_HW_ODD
        else
        {
            global_trans(in_off, out_off, p_blck, n, c, in, out);
        }
#endif

        // TODO: support y=alpha*x+beta*y
    }

#if USE_ALPHA
    (void)alpha;
#endif
#if USE_BETA
    (void)beta;
#endif
}
