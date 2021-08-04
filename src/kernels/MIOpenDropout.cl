/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

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

#ifndef RUN_FORWARD
#define RUN_FORWARD 0
#endif

#ifndef RUN_INIT_PRNG
#define RUN_INIT_PRNG 0
#endif

#if RUN_INIT_PRNG
#include "precalc_xorwow_skipahead_matrices_kernel.h"
#include "precalc_xorwow_skipahead_sequence_matrices_kernel.h"
#endif

typedef struct xorwowStates
{
    // Xorshift values (160 bits)
    uint x;
    uint y;
    uint z;
    uint w;
    uint v;

    // Weyl sequence value
    uint d;
} xorwowStates;

typedef xorwowStates prngStates;

uint xorwow_lite_next(prngStates* cur_state)
{
    const uint t = cur_state->x ^ (cur_state->x >> 2);
    cur_state->x = cur_state->y;
    cur_state->y = cur_state->z;
    cur_state->z = cur_state->w;
    cur_state->w = cur_state->v;
    cur_state->v = (cur_state->v ^ (cur_state->v << 4)) ^ (t ^ (t << 1));

    cur_state->d += 362437;

    return cur_state->d + cur_state->v;
}

#define ROCRAND_2POW32_INV (2.3283064e-10f)

float uniform_distribution(uint v) { return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV); }

#if RUN_INIT_PRNG
void copy_const_arr(uint* dst, constant uint* src, const int arr_size)
{
    for(int i = 0; i < arr_size; i++)
    {
        dst[i] = src[i];
    }
}

void copy_arr(uint* dst, const uint* src, const int arr_size)
{
    for(int i = 0; i < arr_size; i++)
    {
        dst[i] = src[i];
    }
}

void mat_vec(const uint* matrix, uint* vector)
{
    uint result[XORWOW_DIM] = {0};
    for(int i = 0; i < XORWOW_DIM; i++)
    {
        for(int j = 0; j < XORWOW_BITS; j++)
        {
            if(vector[i] & (1 << j))
            {
                for(int k = 0; k < XORWOW_DIM; k++)
                {
                    result[k] ^= matrix[XORWOW_DIM * (i * XORWOW_BITS + j) + k];
                }
            }
        }
    }
    copy_arr(vector, result, XORWOW_DIM);
}

void mat_mat(uint* matrixA, const uint* matrixB)
{
    for(int i = 0; i < XORWOW_DIM * XORWOW_BITS; i++)
    {
        mat_vec(matrixB, matrixA + i * XORWOW_DIM);
    }
}

void mat_identity(uint* matrix)
{
    for(int i = 0; i < XORWOW_DIM; i++)
    {
        for(int j = 0; j < XORWOW_BITS; j++)
        {
            for(int k = 0; k < XORWOW_DIM; k++)
            {
                matrix[(i * XORWOW_BITS + j) * XORWOW_DIM + k] = ((i == k) ? (1 << j) : 0);
            }
        }
    }
}

void mat_pow(uint* matrixP, const uint* matrix, unsigned long long power)
{
    mat_identity(matrixP);

    uint matrixA[XORWOW_PRECALC_MATRICES_SZ];
    uint matrixB[XORWOW_PRECALC_MATRICES_SZ];
    copy_arr(matrixA, matrix, XORWOW_PRECALC_MATRICES_SZ);
    while(power)
    {
        if(power & 1)
        {
            mat_mat(matrixP, matrixA);
        }

        copy_arr(matrixB, matrixA, XORWOW_PRECALC_MATRICES_SZ);
        mat_mat(matrixA, matrixB);
        power >>= 1;
    }
}

void xorwow_skipahead(unsigned long long skp,
                      prngStates* state,
                      constant uint
                          skipahead_mat[XORWOW_PRECALC_MATRICES_NUM][XORWOW_PRECALC_MATRICES_SZ])
{
    uint xor_vec[XORWOW_DIM];
    uint* p = &(state->x);
    for(int i = 0; i < XORWOW_DIM; i++)
    {
        xor_vec[i] = *(p + i);
    }

    uint mat_idx = 0;
    while(skp
#if(XORWOW_PRECALC_MATRICES_NUM * XORWOW_JUMP_LOG2) < 64
          && mat_idx < XORWOW_PRECALC_MATRICES_NUM
#endif
    )
    {
        uint mat[XORWOW_PRECALC_MATRICES_SZ];
        copy_const_arr(mat, skipahead_mat[mat_idx], XORWOW_PRECALC_MATRICES_SZ);

        if(skp & XORWOW_JUMP_LOG2_MASK)
        {
            mat_vec(mat, xor_vec);
        }
        skp >>= XORWOW_JUMP_LOG2;
        mat_idx++;
    }

#if(XORWOW_PRECALC_MATRICES_NUM * XORWOW_JUMP_LOG2) < 64
    if(skp)
    {
        uint matrixA[XORWOW_PRECALC_MATRICES_SZ], matrixB[XORWOW_PRECALC_MATRICES_SZ];
        copy_const_arr(
            matrixA, skipahead_mat[XORWOW_PRECALC_MATRICES_NUM - 1], XORWOW_PRECALC_MATRICES_SZ);

        while(skp)
        {
            mat_pow(matrixB, matrixA, 1ULL << XORWOW_JUMP_LOG2);
            copy_arr(matrixA, matrixB, XORWOW_PRECALC_MATRICES_SZ);

            if(skp & XORWOW_JUMP_LOG2_MASK)
            {
                mat_vec(matrixA, xor_vec);
            }
            skp >>= XORWOW_JUMP_LOG2;
        }
    }
#endif

    for(int i = 0; i < XORWOW_DIM; i++)
    {
        *(p + i) = xor_vec[i];
    }
}

void xorwow_lite_init(prngStates* cur_state,
                      const unsigned long long seed,
                      const unsigned long long subsequence,
                      const unsigned long long offset)
{
    cur_state->x = 123456789;
    cur_state->y = 362436069;
    cur_state->z = 521288629;
    cur_state->w = 88675123;
    cur_state->v = 5783321;

    cur_state->d = 6615241;

    // Adopt constants choice of rocRAND (https://github.com/ROCmSoftwarePlatform/rocRAND)
    const uint s0 = (uint)(seed) ^ 0x2c7f967fU;
    const uint s1 = (uint)(seed >> 32) ^ 0xa03697cbU;
    const uint t0 = 1228688033 * s0;
    const uint t1 = 2073658381 * s1;
    cur_state->x += t0;
    cur_state->y ^= t0;
    cur_state->z += t1;
    cur_state->w ^= t1;
    cur_state->v += t0;
    cur_state->d += t1 + t0;

    xorwow_skipahead(subsequence, cur_state, precalc_xorwow_skipahead_sequence_matrices);

    xorwow_skipahead(offset, cur_state, precalc_xorwow_skipahead_matrices);
    cur_state->d += (uint)(offset)*362437;
}

__kernel void InitKernelState(__global prngStates* state)
{
    for(uint gid = get_global_id(0); gid < STATES_NUM; gid += get_global_size(0))
    {
        prngStates state_gid;
        xorwow_lite_init(&state_gid,
                         (unsigned long long)PRNG_SEED,
                         (unsigned long long)gid,
                         (unsigned long long)0);

        *((__global prngStates*)(state + gid)) = state_gid;
    }
}
#endif

#if !RUN_INIT_PRNG
#ifndef USE_MASK
#define USE_MASK 0
#endif

#ifndef USE_RSVSP
#define USE_RSVSP 0
#endif

#ifndef USE_PRNG
#define USE_PRNG 0
#endif

__kernel void
#if RUN_FORWARD
DropoutForward(
#if USE_MASK
    UNUSED
#endif
#else
DropoutBackward(
#if !USE_PRNG
    UNUSED
#endif
#endif
    const __global prngStates* state,
    const float dropout,
    const float scale,
    const int dim1,
    const int dim2,
    const int dim3,
    const int dim4,
#if !RUN_FORWARD
    const
#endif
    __global _FLOAT* y,
    const int out_str0,
    const int out_str1,
    const int out_str2,
    const int out_str3,
#if RUN_FORWARD
    const
#endif
    __global _FLOAT* x,
    const int in_str0,
    const int in_str1,
    const int in_str2,
    const int in_str3,
#if(RUN_FORWARD && !USE_RSVSP && !USE_MASK) || (!RUN_FORWARD && USE_PRNG)
    UNUSED
#endif
        __global uchar* reserveSpace,
    const uint total_work,
    const uint in_offset,
    const uint out_offset,
#if(RUN_FORWARD && !USE_RSVSP && !USE_MASK) || (!RUN_FORWARD && USE_PRNG)
    UNUSED
#endif
    const uint rsvsp_offset)
{
    _FLOAT dat_blk[RD_BLCK];
    uchar is_kept[RD_BLCK];
#if(RUN_FORWARD && !USE_MASK) || (!RUN_FORWARD && USE_PRNG)
    uint sid = get_global_id(0);
    prngStates cur_state;
    cur_state = *((__global prngStates*)(state + sid));
#endif

    for(uint gid = get_global_id(0); gid < total_work; gid += get_global_size(0))
    {
        uint i0    = gid / dim1 / dim2 / dim3 / dim4;
        uint i1    = (gid / dim2 / dim3 / dim4) % dim1;
        uint i2    = (gid / dim3 / dim4) % dim2;
        uint i3    = (gid / dim4) % dim3;
        uint i4    = gid % dim4;
        uint i4_rd = i4 / RD_BLCK;

        uint x_idx = i0 * in_str0 + i1 * in_str1 + i2 * in_str2 + i3 * in_str3 + i4_rd * RD_BLCK;
        uint y_idx =
            i0 * out_str0 + i1 * out_str1 + i2 * out_str2 + i3 * out_str3 + i4_rd * RD_BLCK;

        *((READ_DAT_TYPE*)dat_blk) = *((const global READ_DAT_TYPE*)(
#if RUN_FORWARD
            x + in_offset + x_idx
#else
            y + out_offset + y_idx
#endif
            ));
#if(RUN_FORWARD && !USE_MASK) || (!RUN_FORWARD && USE_PRNG)
        for(int i = 0; i < RD_BLCK; ++i)
        {
            is_kept[i] = (uchar)(uniform_distribution(xorwow_lite_next(&cur_state)) > dropout);
        }
#if RUN_FORWARD && USE_RSVSP
        *((global READ_BOOL_TYPE*)(reserveSpace + rsvsp_offset + gid - i4 + i4_rd * RD_BLCK)) =
            *((READ_BOOL_TYPE*)is_kept);
#endif
#else
        *((READ_BOOL_TYPE*)is_kept) = *((const global READ_BOOL_TYPE*)(reserveSpace + rsvsp_offset +
                                                                       gid - i4 + i4_rd * RD_BLCK));
#endif
        for(int i = 0; i < RD_BLCK; ++i)
        {
            dat_blk[i] = (bool)(is_kept[i]) ? dat_blk[i] * (_FLOAT)scale : (_FLOAT)0;
        }

        *((global READ_DAT_TYPE*)(
#if RUN_FORWARD
            y + out_offset + y_idx
#else
            x + in_offset + x_idx
#endif
            )) = *((READ_DAT_TYPE*)dat_blk);
    }
    (void)dropout;
}
#endif
