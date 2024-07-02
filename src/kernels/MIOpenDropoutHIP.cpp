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

// Workaround to overcome redefinition errors while including rocrand header files directly
#include "miopen_rocrand.hpp"

#ifndef MIOPEN_USE_FP32
#define MIOPEN_USE_FP32 0
#endif

#ifndef MIOPEN_USE_FP16
#define MIOPEN_USE_FP16 0
#endif

#if MIOPEN_USE_FP16 == 1
#define _FLOAT half
#endif
#if MIOPEN_USE_FP32 == 1
#define _FLOAT float
#endif

#ifndef RUN_FORWARD
#define RUN_FORWARD 0
#endif

#ifndef RUN_INIT_PRNG
#define RUN_INIT_PRNG 0
#endif

#if RUN_INIT_PRNG // Initialize PRNG
/**
 * @brief Initializes the kernel state.
 *
 * This function initializes the kernel state by assigning a random state to each element in the
 * state array.
 *
 * @param state Pointer to the array of rocrand_state_xorwow structures representing the kernel
 * state.
 * @param prng_seed The seed value for the pseudo-random number generator.
 * @param states_num The number of elements in the state array.
 */
extern "C" __global__ void
InitKernelStateHIP(rocrand_state_xorwow* state, ulong prng_seed, ulong states_num)
{
    // Get the index of the current element
    uint index  = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;

    for(uint gid = index; gid < states_num; gid += stride)
    {
        rocrand_state_xorwow state_gid;
        rocrand_init(prng_seed, gid, 0ULL, &state_gid);
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

// Dropout Forward
/**
 * @brief Data is loaded into registers from the input.
 * The mask is either generated and saved in the reserved space, or an existing mask is loaded into
 * registers from the reserved space. The loaded mask is applied to the data in the registers. The
 * processed data is then written to the output. Reading and writing of input data can be performed
 * as vectorized operations, with the extent of vectorization determined by `#RD_BLOCK`.
 */
template <typename F, typename T, typename B, bool MASK = false, bool RSVSP = false>
__forceinline__ __device__ void dropoutfw(const rocrand_state_xorwow* state,
                                          float dropout,
                                          float scale,
                                          int dim1,
                                          int dim2,
                                          int dim3,
                                          int dim4,
                                          F* y,
                                          int out_str0,
                                          int out_str1,
                                          int out_str2,
                                          int out_str3,
                                          const F* x,
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
    F dat_blk[RD_BLCK];     // Register space to read the input data
    uchar is_kept[RD_BLCK]; // Register space to store the mask for the dropout

    uint sid = threadIdx.x + blockIdx.x * blockDim.x;
    rocrand_state_xorwow cur_state; // Read the state of the current thread
    cur_state = state[sid];

    for(uint gid = threadIdx.x + blockIdx.x * blockDim.x; gid < total_work;
        gid += blockDim.x * gridDim.x)
    {
        uint i0    = gid / (dim1 * dim2 * dim3 * dim4);
        uint i1    = (gid / (dim2 * dim3 * dim4)) % dim1;
        uint i2    = (gid / (dim3 * dim4)) % dim2;
        uint i3    = (gid / dim4) % dim3;
        uint i4    = gid % dim4;
        uint i4_rd = i4 / RD_BLCK;

        uint x_idx = i0 * in_str0 + i1 * in_str1 + i2 * in_str2 + i3 * in_str3 +
                     i4_rd * RD_BLCK; // Calculate the index of the input tensor
        uint y_idx = i0 * out_str0 + i1 * out_str1 + i2 * out_str2 + i3 * out_str3 +
                     i4_rd * RD_BLCK; // Calculate the index of the output tensor

        *(reinterpret_cast<T*>(dat_blk)) = *(reinterpret_cast<const T*>(
            x + in_offset + x_idx)); // Read RD_BLCK number of _FLOAT data from the input tensor

        if constexpr(!MASK) // If MASK is not enabled then generate the mask for dropout
        {
            for(int i = 0; i < RD_BLCK; ++i)
            {
                is_kept[i] =
                    static_cast<uchar>(prng::xorwow_uniform(&cur_state) >
                                       dropout); // Generate a random number and compare it with the
                                                 // dropout probability to generate the mask
            }

            if constexpr(RSVSP) // If RSVSP is enabled then store the mask by writing RD_BLCK number
                                // of mask elements to the reserveSpace
            {
                *(reinterpret_cast<B*>(reserveSpace + rsvsp_offset + gid - i4 + i4_rd * RD_BLCK)) =
                    *(reinterpret_cast<B*>(is_kept));
            }
        }
        else
        { // If MASK is enabled then read the mask from the reserveSpace
            *(reinterpret_cast<B*>(is_kept)) = *(reinterpret_cast<const B*>(
                reserveSpace + rsvsp_offset + gid - i4 + i4_rd * RD_BLCK));
        }
        // Apply the mask to the data and scale it with the scale factor.
        for(int i = 0; i < RD_BLCK; ++i)
        {
            dat_blk[i] = static_cast<bool>(is_kept[i]) ? dat_blk[i] * (F)scale : (F)0;
        }
        // Write RD_BLCK number of _FLOAT data to the output tensor
        *(reinterpret_cast<T*>(y + out_offset + y_idx)) = *(reinterpret_cast<T*>(dat_blk));
    }
}

extern "C" __global__ void DropoutFW(const rocrand_state_xorwow* state,
                                     float dropout,
                                     float scale,
                                     int dim1,
                                     int dim2,
                                     int dim3,
                                     int dim4,
                                     _FLOAT* y,
                                     int out_str0,
                                     int out_str1,
                                     int out_str2,
                                     int out_str3,
                                     const _FLOAT* x,
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

    dropoutfw<_FLOAT, READ_DAT_TYPE, READ_BOOL_TYPE, USE_MASK, USE_RSVSP>(state,
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
#if !RUN_FORWARD

#ifndef USE_PRNG
#define USE_PRNG 0
#endif

/**
 * @brief Applies dropout backward operation to the input data.
 * The mask is either generated or an existing mask is loaded into
 * registers from the reserveSpace. The loaded mask is applied to the data in the registers. The
 * processed data is then written to the output. Reading and writing of input data can be performed
 * as vectorized operations, with the extent of vectorization determined by #RD_BLOCK.
 */
template <typename F, typename T, typename B, bool PRNG = false>
__forceinline__ __device__ void dropoutbw(const rocrand_state_xorwow* state,
                                          float dropout,
                                          float scale,
                                          int dim1,
                                          int dim2,
                                          int dim3,
                                          int dim4,
                                          const F* y,
                                          int out_str0,
                                          int out_str1,
                                          int out_str2,
                                          int out_str3,
                                          F* x,
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
    F dat_blk[RD_BLCK];     // Read in the input data into the register space
    uchar is_kept[RD_BLCK]; // is_kept holds the mask for the dropout

    uint sid = threadIdx.x + blockIdx.x * blockDim.x;
    rocrand_state_xorwow cur_state;
    cur_state = state[sid];

    for(uint gid = threadIdx.x + blockIdx.x * blockDim.x; gid < total_work;
        gid += blockDim.x * gridDim.x)
    {
        uint i0    = gid / (dim1 * dim2 * dim3 * dim4);
        uint i1    = (gid / (dim2 * dim3 * dim4)) % dim1;
        uint i2    = (gid / (dim3 * dim4)) % dim2;
        uint i3    = (gid / dim4) % dim3;
        uint i4    = gid % dim4;
        uint i4_rd = i4 / RD_BLCK;

        uint x_idx = i0 * in_str0 + i1 * in_str1 + i2 * in_str2 + i3 * in_str3 +
                     i4_rd * RD_BLCK; // Calculate the index of the output tensor
        uint y_idx = i0 * out_str0 + i1 * out_str1 + i2 * out_str2 + i3 * out_str3 +
                     i4_rd * RD_BLCK; // Calculate the index of the input tensor

        // Read RD_BLCK number of _FLOAT data from y and store it in dat_blk
        *(reinterpret_cast<T*>(dat_blk)) = *(reinterpret_cast<const T*>(y + out_offset + y_idx));

        // If PRNG is enabled then generate the mask for the dropout
        if constexpr(PRNG)
        {
            // Generate mask for RD_BLCK number of elements when it is a vectorized read.
            for(int i = 0; i < RD_BLCK; ++i)
            {

                is_kept[i] = static_cast<uchar>(prng::xorwow_uniform(&cur_state) > dropout);
            }
        }
        else
        {
            // If PRNG is not enabled, then do a vectorized read of the mask from the reserveSpace
            *(reinterpret_cast<B*>(is_kept)) = *(reinterpret_cast<const B*>(
                reserveSpace + rsvsp_offset + gid - i4 + i4_rd * RD_BLCK));
        }

        // Iterate over the RD_BLCK number of elements and apply the mask to the data
        for(int i = 0; i < RD_BLCK; ++i)
        {
            // If the element is retained then scale it with the scale factor
            dat_blk[i] = static_cast<bool>(is_kept[i]) ? dat_blk[i] * (F)scale : (F)0;
        }

        // Write RD_BLCK number of _FLOAT elements to the output in case of a vectorized write
        *(reinterpret_cast<T*>(x + in_offset + x_idx)) = *(reinterpret_cast<T*>(dat_blk));
    }
}

extern "C" __global__ void DropoutBW(const rocrand_state_xorwow* state,
                                     float dropout,
                                     float scale,
                                     int dim1,
                                     int dim2,
                                     int dim3,
                                     int dim4,
                                     const _FLOAT* y,
                                     int out_str0,
                                     int out_str1,
                                     int out_str2,
                                     int out_str3,
                                     _FLOAT* x,
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

    dropoutbw<_FLOAT, READ_DAT_TYPE, READ_BOOL_TYPE, USE_PRNG>(state,
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