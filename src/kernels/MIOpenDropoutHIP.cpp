/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#ifndef RUN_FORWARD
#define RUN_FORWARD 0
#endif

#ifndef RUN_INIT_PRNG
#define RUN_INIT_PRNG 0
#endif

#if RUN_INIT_PRNG
#include "precalc_xorwow_skipahead_matrices_kernel_hip.hpp"
#include "precalc_xorwow_skipahead_sequence_matrices_kernel_hip.hpp"
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

__device__ uint xorwow_lite_next(prngStates* cur_state)
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

__device__ float uniform_distribution(uint v) { return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV); }

#if RUN_INIT_PRNG
__device__ void copy_const_arr(uint* dst, uint* src, const int arr_size)
{
    for(int i = 0; i < arr_size; i++)
    {
        dst[i] = src[i];
    }
}

__device__ void copy_arr(uint* dst, const uint* src, const int arr_size)
{
    for(int i = 0; i < arr_size; i++)
    {
        dst[i] = src[i];
    }
}

__device__ void mat_vec(const uint* matrix, uint* vector)
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

__device__ void mat_mat(uint* matrixA, const uint* matrixB)
{
    for(int i = 0; i < XORWOW_DIM * XORWOW_BITS; i++)
    {
        mat_vec(matrixB, matrixA + i * XORWOW_DIM);
    }
}

__device__ void mat_identity(uint* matrix)
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

__device__ void mat_pow(uint* matrixP, const uint* matrix, unsigned long long power)
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

__device__ void xorwow_skipahead(unsigned long long skp,
                      prngStates* state,
                       uint skipahead_mat[XORWOW_PRECALC_MATRICES_NUM][XORWOW_PRECALC_MATRICES_SZ])
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

__device__ void xorwow_lite_init(prngStates* cur_state,
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

    // Adopt constants choice of rocRAND (https://github.com/ROCm/rocRAND)
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

/* 
    This kernel sets up the states based on the seed, offset and two precalculated matrices
    1) precalc_xorwow_skipahead_sequence_matrices
    2) precalc_xorwow_skipahead_matrices
*/
extern "C" __global__ void InitKernelStateHIP(prngStates* state, ulong prng_seed, ulong states_num)
{
    // Get the index of the current element
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;

    for(uint gid = index; gid < states_num; gid += stride)
    {
        prngStates state_gid;
        xorwow_lite_init(&state_gid,
                         prng_seed,
                         gid,
                         0ULL);

        state[gid] = state_gid;
        
    }
    
}
#endif

#if !RUN_INIT_PRNG

#if RUN_FORWARD
#ifndef USE_MASK
#define USE_MASK 0
#endif

#ifndef USE_RSVSP
#define USE_RSVSP 0
#endif
#endif

#if !RUN_FORWARD
#ifndef USE_PRNG
#define USE_PRNG 0
#endif
#endif

#if RUN_FORWARD
template <typename T, typename B, bool MASK=false, bool RSVSP=false>
__forceinline__ __device__ void dropoutfw(
    const prngStates* state,
    float dropout,
    float scale,
    int dim1,
    int dim2,
    int dim3,
    int dim4,
    T* y,
    int out_str0,
    int out_str1,
    int out_str2,
    int out_str3,
    const T* x,
    int in_str0,
    int in_str1,
    int in_str2,
    int in_str3,
    uchar* reserveSpace,
    uint total_work,
    uint in_offset,
    uint out_offset,
    uint rsvsp_offset)
{
    T dat_blk[RD_BLCK];
    uchar is_kept[RD_BLCK];

    uint sid = threadIdx.x + blockIdx.x * blockDim.x;
    prngStates cur_state;
    cur_state = state[sid];

    for(uint gid = threadIdx.x + blockIdx.x * blockDim.x; gid < total_work; gid += blockDim.x * gridDim.x)
    {
        uint i0    = gid / (dim1 * dim2 * dim3 * dim4);
        uint i1    = (gid / (dim2 * dim3 * dim4)) % dim1;
        uint i2    = (gid / (dim3 * dim4)) % dim2;
        uint i3    = (gid / dim4) % dim3;
        uint i4    = gid % dim4;
        uint i4_rd = i4 / RD_BLCK;

        uint x_idx = i0 * in_str0 + i1 * in_str1 + i2 * in_str2 + i3 * in_str3 + i4_rd * RD_BLCK;
        uint y_idx = i0 * out_str0 + i1 * out_str1 + i2 * out_str2 + i3 * out_str3 + i4_rd * RD_BLCK;

        dat_blk[0] = x[in_offset + x_idx];

        if constexpr (!MASK)
        {
            for(int i = 0; i < RD_BLCK; ++i)
            {
                is_kept[i] = static_cast<uchar>(uniform_distribution(xorwow_lite_next(&cur_state)) > dropout);
            }

            if constexpr (RSVSP)
            {
                reserveSpace[rsvsp_offset + gid - i4 + i4_rd * RD_BLCK] = is_kept[0];
            }
        } else {
            is_kept[0] = reserveSpace[rsvsp_offset + gid - i4 + i4_rd * RD_BLCK];
        }

        for(int i = 0; i < RD_BLCK; ++i)
        {
            dat_blk[i] = is_kept[i] ? dat_blk[i] * (T)scale : (T)0;
        }

        y[out_offset + y_idx] = dat_blk[0];
    }

}

// Repalce the READ_DAT_TYPE with INPUT_TYPE and OUTPUT_TYPE
extern "C" __global__ void DropoutFW(
    const prngStates* state,
    float dropout,
    float scale,
    int dim1,
    int dim2,
    int dim3,
    int dim4,
    READ_DAT_TYPE* y,
    int out_str0,
    int out_str1,
    int out_str2,
    int out_str3,
    const READ_DAT_TYPE* x,
    int in_str0,
    int in_str1,
    int in_str2,
    int in_str3,
    uchar* reserveSpace,
    uint total_work,
    uint in_offset,
    uint out_offset,
    uint rsvsp_offset
)
{

    dropoutfw<READ_DAT_TYPE, READ_BOOL_TYPE, USE_MASK, USE_RSVSP>(
        state,
        dropout,
        scale,
        dim1,
        dim2,
        dim3,
        dim4,
        y,
        out_str0,
        out_str1,
        out_str2,
        out_str3,
        x,
        in_str0,
        in_str1,
        in_str2,
        in_str3,
        reserveSpace,
        total_work,
        in_offset,
        out_offset,
        rsvsp_offset);

}
#endif


// Dropout Backward
#if !RUN_FORWARD && !RUN_INIT_PRNG
template <typename T, typename B, bool PRNG=false>
__forceinline__ __device__ void dropoutbw(
    const prngStates* state,
    float dropout,
    float scale,
    int dim1,
    int dim2,
    int dim3,
    int dim4,
    const T* y,
    int out_str0,
    int out_str1,
    int out_str2,
    int out_str3,
    T* x,
    int in_str0,
    int in_str1,
    int in_str2,
    int in_str3,
    uchar* reserveSpace,
    uint total_work,
    uint in_offset,
    uint out_offset,
    uint rsvsp_offset)
{
    T dat_blk[RD_BLCK];
    uchar is_kept[RD_BLCK];

    uint sid = threadIdx.x + blockIdx.x * blockDim.x;
    prngStates cur_state;
    cur_state = state[sid];

    for(uint gid = threadIdx.x + blockIdx.x * blockDim.x; gid < total_work; gid += blockDim.x * gridDim.x)
    {
        uint i0    = gid / (dim1 * dim2 * dim3 * dim4);
        uint i1    = (gid / (dim2 * dim3 * dim4)) % dim1;
        uint i2    = (gid / (dim3 * dim4)) % dim2;
        uint i3    = (gid / dim4) % dim3;
        uint i4    = gid % dim4;
        uint i4_rd = i4 / RD_BLCK;

        uint x_idx = i0 * in_str0 + i1 * in_str1 + i2 * in_str2 + i3 * in_str3 + i4_rd * RD_BLCK;
        uint y_idx = i0 * out_str0 + i1 * out_str1 + i2 * out_str2 + i3 * out_str3 + i4_rd * RD_BLCK;

        dat_blk[0] = y[out_offset + y_idx];

        if constexpr (PRNG)
        {
            for(int i = 0; i < RD_BLCK; ++i)
            {
                is_kept[i] = static_cast<uchar>(uniform_distribution(xorwow_lite_next(&cur_state)) > dropout);
            }

        } else {
            is_kept[0] = reserveSpace[rsvsp_offset + gid - i4 + i4_rd * RD_BLCK];
        }

        for(int i = 0; i < RD_BLCK; ++i)
        {
            dat_blk[i] = is_kept[i] ? dat_blk[i] * (T)scale : (T)0;
        }

        x[in_offset + x_idx] = dat_blk[0];
    }

}

// Replace the READ_DAT_TYPE with INPUT_TYPE and OUTPUT_TYPE

extern "C" __global__ void DropoutBW(
    const prngStates* state,
    float dropout,
    float scale,
    int dim1,
    int dim2,
    int dim3,
    int dim4,
    const READ_DAT_TYPE* y,
    int out_str0,
    int out_str1,
    int out_str2,
    int out_str3,
    READ_DAT_TYPE* x,
    int in_str0,
    int in_str1,
    int in_str2,
    int in_str3,
    uchar* reserveSpace,
    uint total_work,
    uint in_offset,
    uint out_offset,
    uint rsvsp_offset
)
{

    dropoutbw<READ_DAT_TYPE, READ_BOOL_TYPE, USE_PRNG>(
        state,
        dropout,
        scale,
        dim1,
        dim2,
        dim3,
        dim4,
        y,
        out_str0,
        out_str1,
        out_str2,
        out_str3,
        x,
        in_str0,
        in_str1,
        in_str2,
        in_str3,
        reserveSpace,
        total_work,
        in_offset,
        out_offset,
        rsvsp_offset);

}
#endif

#endif