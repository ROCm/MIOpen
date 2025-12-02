/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#define UNUSED __attribute__((__unused__))

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "activation_functions.hpp"

#ifdef LITE

/**********************************************************************************************
**********************************************************************************************/

// N - batch size
// C - # of maps
// H - map height
// W - map width
// TENS_LEN = (N*C*H*W);
// RD_BLCK = (TENS_LEN%4==0) ? 4 : (TENS_LEN%3==0)? 3 : (TENS_LEN%2==0)? 2 : 1;
// READ_TYPE = (RD_BLCK==4) ? "float4" : (RD_BLCK == 3) ? "float3" : (RD_BLC==2) ? "float2" :
// "float";
// local size = (256, 1, 1)
// global size = ((TENS_LEN/RD_BLCK), 1, 1)

extern "C" __global__ void MIOpenActiveFwdLite(const FP_TYPE* bot,
                                               FP_TYPE* top,
                                               const int map_size_aligned,
                                               FP_TYPE gamma,
                                               FP_TYPE beta,
                                               FP_TYPE alpha,
                                               const long bot_offset,
                                               const long top_offset)
{
    const unsigned int tid   = blockIdx.x * LOCAL_SIZE + threadIdx.x;
    const unsigned int index = tid * MIOPEN_READ_UNIT;

    if(tid >= map_size_aligned)
        return;

    FP_TYPE data[MIOPEN_READ_UNIT];
    FP_TYPE response[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)data) = *((const MIOPEN_READ_TYPE*)(bot + bot_offset + index));

    ActivationFunction(response, data, gamma, beta, alpha);

    *((MIOPEN_READ_TYPE*)(top + top_offset + index)) = *((MIOPEN_READ_TYPE*)response);
}

/**********************************************************************************************
**********************************************************************************************/

extern "C" __global__ void MIOpenActiveFwd2DLite(const FP_TYPE* bot,
                                                 FP_TYPE* top,
                                                 const int map_size_aligned,
                                                 const int height,
                                                 FP_TYPE gamma,
                                                 FP_TYPE beta,
                                                 FP_TYPE alpha,
                                                 const long bot_offset,
                                                 const long top_offset,
                                                 const uint bot_stride,
                                                 const uint top_stride)
{
    const unsigned int x_id = blockIdx.x * LOCAL_SIZE + threadIdx.x;
    const unsigned int y    = blockIdx.y * blockDim.y + threadIdx.y;

    if(x_id >= map_size_aligned)
        return;

    if(y >= height)
        return;

    size_t bot_index = y * bot_stride + x_id * MIOPEN_READ_UNIT;
    size_t top_index = y * top_stride + x_id * MIOPEN_READ_UNIT;

    FP_TYPE data[MIOPEN_READ_UNIT];
    FP_TYPE response[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)data) = *((const MIOPEN_READ_TYPE*)(bot + bot_offset + bot_index));

    ActivationFunction(response, data, gamma, beta, alpha);

    *((MIOPEN_READ_TYPE*)(top + top_offset + top_index)) = *((MIOPEN_READ_TYPE*)response);
}

/**********************************************************************************************
**********************************************************************************************/

extern "C" __global__ void MIOpenActiveBwdLite(FP_TYPE* bot_diff,
                                               const FP_TYPE* top_diff,
                                               const FP_TYPE* bot,
                                               const FP_TYPE* top,
                                               const int map_size_aligned,
                                               FP_TYPE diff_scale,
                                               FP_TYPE gamma,
                                               FP_TYPE beta,
                                               FP_TYPE alpha,
                                               const long bot_diff_offset,
                                               const long top_diff_offset,
                                               const long bot_offset,
                                               const long top_offset)
{
    const unsigned int tid = blockIdx.x * LOCAL_SIZE + threadIdx.x;
    int index              = tid * MIOPEN_READ_UNIT;

    if(tid >= map_size_aligned)
        return;

    FP_TYPE bot_diff_dat[MIOPEN_READ_UNIT];
    FP_TYPE top_diff_dat[MIOPEN_READ_UNIT];
    FP_TYPE bot_dat[MIOPEN_READ_UNIT];
    FP_TYPE top_dat[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)top_diff_dat) =
        *((const MIOPEN_READ_TYPE*)(top_diff + top_diff_offset + index));
    *((MIOPEN_READ_TYPE*)bot_dat) = *((const MIOPEN_READ_TYPE*)(bot + bot_offset + index));
    *((MIOPEN_READ_TYPE*)top_dat) = *((const MIOPEN_READ_TYPE*)(top + top_offset + index));

    ActivationFunction_Diff(
        bot_diff_dat, top_diff_dat, bot_dat, top_dat, diff_scale, gamma, beta, alpha);

    *((MIOPEN_READ_TYPE*)(bot_diff + bot_diff_offset + index)) = *((MIOPEN_READ_TYPE*)bot_diff_dat);
}

/**********************************************************************************************
**********************************************************************************************/

extern "C" __global__ void MIOpenActiveBwd2DLite(FP_TYPE* bot_diff,
                                                 const FP_TYPE* top_diff,
                                                 const FP_TYPE* bot,
                                                 const FP_TYPE* top,
                                                 const int map_size_aligned,
                                                 const int height,
                                                 FP_TYPE diff_scale,
                                                 FP_TYPE gamma,
                                                 FP_TYPE beta,
                                                 FP_TYPE alpha,
                                                 const long bot_diff_offset,
                                                 const long top_diff_offset,
                                                 const long bot_offset,
                                                 const long top_offset,
                                                 const uint bot_diff_stride,
                                                 const uint top_diff_stride,
                                                 const uint bot_stride,
                                                 const uint top_stride)
{
    const unsigned int x_id = blockIdx.x * LOCAL_SIZE + threadIdx.x;
    const unsigned int y    = blockIdx.y * blockDim.y + threadIdx.y;

    if(x_id >= map_size_aligned)
        return;

    if(y >= height)
        return;

    uint bot_diff_index = y * bot_diff_stride + x_id * MIOPEN_READ_UNIT;
    uint top_diff_index = y * top_diff_stride + x_id * MIOPEN_READ_UNIT;
    uint bot_index      = y * bot_stride + x_id * MIOPEN_READ_UNIT;
    uint top_index      = y * top_stride + x_id * MIOPEN_READ_UNIT;

    FP_TYPE bot_diff_dat[MIOPEN_READ_UNIT];
    FP_TYPE top_diff_dat[MIOPEN_READ_UNIT];
    FP_TYPE bot_dat[MIOPEN_READ_UNIT];
    FP_TYPE top_dat[MIOPEN_READ_UNIT];

    *((MIOPEN_READ_TYPE*)top_diff_dat) =
        *((const MIOPEN_READ_TYPE*)(top_diff + top_diff_offset + top_diff_index));
    *((MIOPEN_READ_TYPE*)bot_dat) = *((const MIOPEN_READ_TYPE*)(bot + bot_offset + bot_index));
    *((MIOPEN_READ_TYPE*)top_dat) = *((const MIOPEN_READ_TYPE*)(top + top_offset + top_index));

    ActivationFunction_Diff(
        bot_diff_dat, top_diff_dat, bot_dat, top_dat, diff_scale, gamma, beta, alpha);

    *((MIOPEN_READ_TYPE*)(bot_diff + bot_diff_offset + bot_diff_index)) =
        *((MIOPEN_READ_TYPE*)bot_diff_dat);
}
/**************************************************************************************************************/

#else

/***************************************************************************************************************/
__launch_bounds__(
    MIOPEN_NRN_GROUP_SZ0* MIOPEN_NRN_GROUP_SZ1* MIOPEN_NRN_GROUP_SZ2) extern "C" __global__
    void MIOpenNeuronFwd(const FP_TYPE* bot,
                         FP_TYPE* top,
                         const int map_size_aligned,
                         FP_TYPE gamma,
                         FP_TYPE beta,
                         FP_TYPE alpha,
                         const long xOffset,
                         const long yOffset)
{
    const unsigned int x = blockIdx.x * MIOPEN_NRN_GROUP_SZ0 + threadIdx.x; // channel x

    if(x >= map_size_aligned)
        return;

    FP_TYPE data[MIOPEN_READ_UNIT];
    FP_TYPE response[MIOPEN_READ_UNIT];
    auto load_element = [&](int i) -> FP_TYPE {
        if constexpr(MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ)
        {
            if constexpr(MIOPEN_N_IN_STRIDE > MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN &&
                         MIOPEN_C_IN != 0 && MIOPEN_H_IN != 0 && MIOPEN_W_IN != 0)
            {
                const int in_hw      = MIOPEN_H_IN * MIOPEN_W_IN;
                const int in_chw     = MIOPEN_C_IN * in_hw;
                const int loc        = x * MIOPEN_READ_UNIT + i;
                const int loc_in_chw = loc % in_chw;
                const int loc_in_hw  = loc_in_chw % in_hw;
                const int n_loc      = loc / in_chw;
                const int c_loc      = loc_in_chw / in_hw;
                const int h_loc      = loc_in_hw / MIOPEN_W_IN;
                const int w_loc      = loc_in_hw % MIOPEN_W_IN;

                return bot[xOffset + n_loc * MIOPEN_N_IN_STRIDE + c_loc * MIOPEN_C_IN_STRIDE +
                           h_loc * MIOPEN_H_IN_STRIDE + w_loc * MIOPEN_W_IN_STRIDE];
            }
        }
        return bot[xOffset + x * MIOPEN_READ_UNIT + i];
    };

    if constexpr(MIOPEN_N_PIXS_OFF > 0)
    {
        if(x == map_size_aligned - 1)
        {
            int i = 0;
#pragma unroll
            for(; i < MIOPEN_N_PIXS_OFF; ++i)
            {
                data[i] = load_element(i);
            }
#pragma unroll
            for(; i < MIOPEN_READ_UNIT; ++i)
            {
                data[i] = (FP_TYPE)0.f;
            }
        }
        else
        {
#pragma unroll
            for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
            {
                data[i] = load_element(i);
            }
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            data[i] = load_element(i);
        }
    }

    ActivationFunction(response, data, gamma, beta, alpha);

    auto store_element = [&](int i, FP_TYPE value) {
        if constexpr(MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ)
        {
            if constexpr(MIOPEN_N_OUT_STRIDE > MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT &&
                         MIOPEN_C_OUT != 0 && MIOPEN_H_OUT != 0 && MIOPEN_W_OUT != 0)
            {
                const int out_hw     = MIOPEN_H_OUT * MIOPEN_W_OUT;
                const int out_chw    = MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT;
                const int loc        = x * MIOPEN_READ_UNIT + i;
                const int loc_in_chw = loc % out_chw;
                const int loc_in_hw  = loc_in_chw % out_hw;
                const int n_loc      = loc / out_chw;
                const int c_loc      = loc_in_chw / out_hw;
                const int h_loc      = loc_in_hw / MIOPEN_W_OUT;
                const int w_loc      = loc_in_hw % MIOPEN_W_OUT;

                top[yOffset + n_loc * MIOPEN_N_OUT_STRIDE + c_loc * MIOPEN_C_OUT_STRIDE +
                    h_loc * MIOPEN_H_OUT_STRIDE + w_loc * MIOPEN_W_OUT_STRIDE] = value;
                return;
            }
        }
        top[yOffset + x * MIOPEN_READ_UNIT + i] = value;
    };

    if constexpr(MIOPEN_N_PIXS_OFF > 0)
    {
        if(x == map_size_aligned - 1)
        {
#pragma unroll
            for(int i = 0; i < MIOPEN_N_PIXS_OFF; ++i)
            {
                store_element(i, response[i]);
            }
            return;
        }
    }

#pragma unroll
    for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
    {
        store_element(i, response[i]);
    }
}

__launch_bounds__(
    MIOPEN_NRN_GROUP_SZ0* MIOPEN_NRN_GROUP_SZ1* MIOPEN_NRN_GROUP_SZ2) extern "C" __global__
    void MIOpenNeuronBwd(FP_TYPE* bot_diff,
                         const FP_TYPE* top_diff,
                         const FP_TYPE* bot_data,
                         const FP_TYPE* top_data,
                         const int map_size_aligned,
                         FP_TYPE diff_scale,
                         FP_TYPE gamma,
                         FP_TYPE beta,
                         FP_TYPE alpha,
                         const long dxOffset,
                         const long dyOffset,
                         const long xOffset,
                         const long yOffset)
{
    const unsigned int x = blockIdx.x * MIOPEN_NRN_GROUP_SZ0 + threadIdx.x;

    if(x >= map_size_aligned)
        return;

    FP_TYPE bot_diff_dat[MIOPEN_READ_UNIT];
    FP_TYPE top_diff_dat[MIOPEN_READ_UNIT];
    FP_TYPE bot_dat[MIOPEN_READ_UNIT];
    FP_TYPE top_dat[MIOPEN_READ_UNIT];

    auto load_top_diff = [&](int i) -> FP_TYPE {
        if constexpr(MIOPEN_N_DOUT_STRIDE > MIOPEN_DOUT_BLOCK_SZ)
        {
            if constexpr(MIOPEN_N_DOUT_STRIDE > MIOPEN_C_DOUT * MIOPEN_H_DOUT * MIOPEN_W_DOUT &&
                         MIOPEN_C_DOUT != 0 && MIOPEN_H_DOUT != 0 && MIOPEN_W_DOUT != 0)
            {
                const int dout_hw    = MIOPEN_H_DOUT * MIOPEN_W_DOUT;
                const int dout_chw   = MIOPEN_C_DOUT * dout_hw;
                const int loc        = x * MIOPEN_READ_UNIT + i;
                const int loc_in_chw = loc % dout_chw;
                const int loc_in_hw  = loc_in_chw % dout_hw;
                const int n_loc      = loc / dout_chw;
                const int c_loc      = loc_in_chw / dout_hw;
                const int h_loc      = loc_in_hw / MIOPEN_W_DOUT;
                const int w_loc      = loc_in_hw % MIOPEN_W_DOUT;

                return top_diff[dyOffset + n_loc * MIOPEN_N_DOUT_STRIDE +
                                c_loc * MIOPEN_C_DOUT_STRIDE + h_loc * MIOPEN_H_DOUT_STRIDE +
                                w_loc * MIOPEN_W_DOUT_STRIDE];
            }
        }
        return top_diff[dyOffset + x * MIOPEN_READ_UNIT + i];
    };

    auto load_bot_data = [&](int i) -> FP_TYPE {
        if constexpr(MIOPEN_N_IN_STRIDE > MIOPEN_IN_BLOCK_SZ)
        {
            if constexpr(MIOPEN_N_IN_STRIDE > MIOPEN_C_IN * MIOPEN_H_IN * MIOPEN_W_IN &&
                         MIOPEN_C_IN != 0 && MIOPEN_H_IN != 0 && MIOPEN_W_IN != 0)
            {
                const int in_hw      = MIOPEN_H_IN * MIOPEN_W_IN;
                const int in_chw     = MIOPEN_C_IN * in_hw;
                const int loc        = x * MIOPEN_READ_UNIT + i;
                const int loc_in_chw = loc % in_chw;
                const int loc_in_hw  = loc_in_chw % in_hw;
                const int n_loc      = loc / in_chw;
                const int c_loc      = loc_in_chw / in_hw;
                const int h_loc      = loc_in_hw / MIOPEN_W_IN;
                const int w_loc      = loc_in_hw % MIOPEN_W_IN;

                return bot_data[xOffset + n_loc * MIOPEN_N_IN_STRIDE + c_loc * MIOPEN_C_IN_STRIDE +
                                h_loc * MIOPEN_H_IN_STRIDE + w_loc * MIOPEN_W_IN_STRIDE];
            }
        }
        return bot_data[xOffset + x * MIOPEN_READ_UNIT + i];
    };

    auto load_top_data = [&](int i) -> FP_TYPE {
        if constexpr(MIOPEN_N_OUT_STRIDE > MIOPEN_OUT_BLOCK_SZ)
        {
            if constexpr(MIOPEN_N_OUT_STRIDE > MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT &&
                         MIOPEN_C_OUT != 0 && MIOPEN_H_OUT != 0 && MIOPEN_W_OUT != 0)
            {
                const int out_hw     = MIOPEN_H_OUT * MIOPEN_W_OUT;
                const int out_chw    = MIOPEN_C_OUT * MIOPEN_H_OUT * MIOPEN_W_OUT;
                const int loc        = x * MIOPEN_READ_UNIT + i;
                const int loc_in_chw = loc % out_chw;
                const int loc_in_hw  = loc_in_chw % out_hw;
                const int n_loc      = loc / out_chw;
                const int c_loc      = loc_in_chw / out_hw;
                const int h_loc      = loc_in_hw / MIOPEN_W_OUT;
                const int w_loc      = loc_in_hw % MIOPEN_W_OUT;

                return top_data[yOffset + n_loc * MIOPEN_N_OUT_STRIDE +
                                c_loc * MIOPEN_C_OUT_STRIDE + h_loc * MIOPEN_H_OUT_STRIDE +
                                w_loc * MIOPEN_W_OUT_STRIDE];
            }
        }
        return top_data[yOffset + x * MIOPEN_READ_UNIT + i];
    };

    if constexpr(MIOPEN_N_PIXS_OFF > 0)
    {
        if(x == map_size_aligned - 1)
        {
            int i = 0;
#pragma unroll
            for(; i < MIOPEN_N_PIXS_OFF; ++i)
            {
                top_diff_dat[i] = load_top_diff(i);
                bot_dat[i]      = load_bot_data(i);
                top_dat[i]      = load_top_data(i);
            }
#pragma unroll
            for(; i < MIOPEN_READ_UNIT; ++i)
            {
                top_diff_dat[i] = (FP_TYPE)0.f;
                bot_dat[i]      = (FP_TYPE)0.f;
                top_dat[i]      = (FP_TYPE)0.f;
            }
        }
        else
        {
#pragma unroll
            for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
            {
                top_diff_dat[i] = load_top_diff(i);
                bot_dat[i]      = load_bot_data(i);
                top_dat[i]      = load_top_data(i);
            }
        }
    }
    else
    {
#pragma unroll
        for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            top_diff_dat[i] = load_top_diff(i);
            bot_dat[i]      = load_bot_data(i);
            top_dat[i]      = load_top_data(i);
        }
    }

    ActivationFunction_Diff(
        bot_diff_dat, top_diff_dat, bot_dat, top_dat, diff_scale, gamma, beta, alpha);

    auto store_bot_diff = [&](int i, FP_TYPE value) {
        if constexpr(MIOPEN_N_DIN_STRIDE > MIOPEN_DIN_BLOCK_SZ)
        {
            if constexpr(MIOPEN_N_DIN_STRIDE > MIOPEN_C_DIN * MIOPEN_H_DIN * MIOPEN_W_DIN &&
                         MIOPEN_C_DIN != 0 && MIOPEN_H_DIN != 0 && MIOPEN_W_DIN != 0)
            {
                const int din_hw     = MIOPEN_H_DIN * MIOPEN_W_DIN;
                const int din_chw    = MIOPEN_C_DIN * din_hw;
                const int loc        = x * MIOPEN_READ_UNIT + i;
                const int loc_in_chw = loc % din_chw;
                const int loc_in_hw  = loc_in_chw % din_hw;
                const int n_loc      = loc / din_chw;
                const int c_loc      = loc_in_chw / din_hw;
                const int h_loc      = loc_in_hw / MIOPEN_W_DIN;
                const int w_loc      = loc_in_hw % MIOPEN_W_DIN;

                bot_diff[dxOffset + n_loc * MIOPEN_N_DIN_STRIDE + c_loc * MIOPEN_C_DIN_STRIDE +
                         h_loc * MIOPEN_H_DIN_STRIDE + w_loc * MIOPEN_W_DIN_STRIDE] = value;
                return;
            }
        }
        bot_diff[dxOffset + x * MIOPEN_READ_UNIT + i] = value;
    };

    if constexpr(MIOPEN_N_PIXS_OFF > 0)
    {
        if(x == map_size_aligned - 1)
        {
#pragma unroll
            for(int i = 0; i < MIOPEN_N_PIXS_OFF; ++i)
            {
                store_bot_diff(i, bot_diff_dat[i]);
            }
            return;
        }
    }

#pragma unroll
    for(int i = 0; i < MIOPEN_READ_UNIT; ++i)
    {
        store_bot_diff(i, bot_diff_dat[i]);
    }
}

#endif // #ifdef LITE
