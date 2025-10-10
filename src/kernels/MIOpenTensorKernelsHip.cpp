/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <hip/hip_bfloat16.h>
#endif

#include "miopen_cstdint.hpp"

template <typename T>
__device__ T miopenAdd(T a, T b)
{
    return a + b;
}

template <typename T>
__device__ T miopenMul(T a, T b)
{
    return a * b;
}

template <typename T>
__device__ T miopenMax(T a, T b)
{
    return ((a > b) ? a : b);
}

template <typename T>
__device__ T miopenMin(T a, T b)
{
    return ((a < b) ? a : b);
}

#ifdef USE_1D_TENSOR_GENERIC
// N
extern "C" __global__ void Op1dTensorGeneric(const MIOPEN_TYPE* a,
                                             const MIOPEN_TYPE* b,
                                             MIOPEN_TYPE* c,
                                             const uint64_t Aoffset,
                                             const uint64_t Boffset,
                                             const uint64_t Coffset,
                                             const DIM_TYPE a_nstride,
                                             const DIM_TYPE b_nstride,
                                             const DIM_TYPE c_nstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const DIM_TYPE total_work,
                                             const bool use_beta)
{
    const MIOPEN_TYPE* a_off = a + Aoffset;
    const MIOPEN_TYPE* b_off = b + Boffset;
    MIOPEN_TYPE* c_off       = c + Coffset;

    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    auto a_ptr     = a_off + gid * a_nstride;
    auto b_ptr     = b_off + gid * b_nstride;
    auto c_ptr     = c_off + gid * c_nstride;

    const auto step   = gridDim.x * blockDim.x;
    const auto a_step = step * a_nstride;
    const auto b_step = step * b_nstride;
    const auto c_step = step * c_nstride;

    const auto c_end = c_off + total_work * c_nstride;
    while(c_ptr < c_end)
    {
        const auto res = MIOPEN_TENSOR_OP(a_ptr[0] * alpha0, b_ptr[0] * alpha1);
        c_ptr[0]       = use_beta ? c_ptr[0] * beta + res : res;

        a_ptr += a_step;
        b_ptr += b_step;
        c_ptr += c_step;
    }
}

#endif

#ifdef USE_2D_TENSOR_GENERIC
// NC
extern "C" __global__ void Op2dTensorGeneric(const MIOPEN_TYPE* a,
                                             const MIOPEN_TYPE* b,
                                             MIOPEN_TYPE* c,
                                             const uint64_t Aoffset,
                                             const uint64_t Boffset,
                                             const uint64_t Coffset,
                                             const DIM_TYPE b_c,
                                             const DIM_TYPE c_c,
                                             const DIM_TYPE a_nstride,
                                             const DIM_TYPE a_cstride,
                                             const DIM_TYPE b_nstride,
                                             const DIM_TYPE b_cstride,
                                             const DIM_TYPE c_nstride,
                                             const DIM_TYPE c_cstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const DIM_TYPE total_work,
                                             const bool use_beta)
{
    const MIOPEN_TYPE* a_off = a + Aoffset;
    const MIOPEN_TYPE* b_off = b + Boffset;
    MIOPEN_TYPE* c_off       = c + Coffset;

    auto gid          = blockIdx.x * blockDim.x + threadIdx.x;
    const auto* a_ptr = a_off + (gid / c_c) * a_nstride + (gid % c_c) * a_cstride;
    auto* c_ptr       = c_off + (gid / c_c) * c_nstride + (gid % c_c) * c_cstride;

    const auto step   = gridDim.x * blockDim.x;
    const auto a_step = (step / c_c) * a_nstride + (step % c_c) * a_cstride;
    const auto c_step = (step / c_c) * c_nstride + (step % c_c) * c_cstride;

    const auto c_end = c_off + total_work * c_nstride;
    while(c_ptr < c_end)
    {
        const auto* b_ptr = b_off;
        if(b_nstride != 0)
            b_ptr += (gid / b_c) * b_nstride;

        if(b_cstride != 0)
            b_ptr += (gid % b_c) * b_cstride;

        auto b_val = *b_ptr;
        auto a_val = *a_ptr;
        auto c_val = use_beta ? *c_ptr : static_cast<MIOPEN_TYPE>(0);
        *c_ptr     = MIOPEN_TENSOR_OP(b_val * alpha1, a_val * alpha0) + c_val * beta;

        a_ptr += a_step;
        c_ptr += c_step;
        gid += step;
    }
}

#endif

#ifdef USE_2D_TENSOR_SQUASH
extern "C" __global__ void Op2dTensorSquash(const MIOPEN_TYPE* a,
                                            const MIOPEN_TYPE* b,
                                            const int b_c,
                                            const int b_nstride,
                                            MIOPEN_TYPE* c,
                                            const MIOPEN_TYPE alpha0,
                                            const MIOPEN_TYPE alpha1,
                                            const MIOPEN_TYPE beta,
                                            const long Aoffset,
                                            const long Boffset,
                                            const long Coffset,
                                            const long total_work,
                                            const int use_apl0,
                                            const int use_apl1,
                                            const int use_bet)
{
    MIOPEN_TYPE a_dat[RD_BLCK];
    MIOPEN_TYPE b_dat1[RD_BLCK];
    MIOPEN_TYPE b_dat2[RD_BLCK];
    MIOPEN_TYPE b_dat3[RD_BLCK];
    MIOPEN_TYPE b_dat4[RD_BLCK];
    MIOPEN_TYPE b_dat5[RD_BLCK];
    MIOPEN_TYPE b_dat6[RD_BLCK];
    MIOPEN_TYPE b_dat7[RD_BLCK];
    MIOPEN_TYPE b_dat8[RD_BLCK];
    MIOPEN_TYPE b_dat9[RD_BLCK];
    MIOPEN_TYPE b_dat10[RD_BLCK];
    MIOPEN_TYPE b_dat11[RD_BLCK];
    MIOPEN_TYPE b_dat12[RD_BLCK];
    MIOPEN_TYPE b_dat13[RD_BLCK];
    MIOPEN_TYPE b_dat14[RD_BLCK];
    MIOPEN_TYPE b_dat15[RD_BLCK];
    MIOPEN_TYPE b_dat16[RD_BLCK];
    MIOPEN_TYPE c_dat[RD_BLCK];
    int g_RD_BLCK;

    for(int i = 0; i < RD_BLCK; ++i)
    {
        b_dat1[i]  = (MIOPEN_TYPE)0;
        b_dat2[i]  = (MIOPEN_TYPE)0;
        b_dat3[i]  = (MIOPEN_TYPE)0;
        b_dat4[i]  = (MIOPEN_TYPE)0;
        b_dat5[i]  = (MIOPEN_TYPE)0;
        b_dat6[i]  = (MIOPEN_TYPE)0;
        b_dat7[i]  = (MIOPEN_TYPE)0;
        b_dat8[i]  = (MIOPEN_TYPE)0;
        b_dat9[i]  = (MIOPEN_TYPE)0;
        b_dat10[i] = (MIOPEN_TYPE)0;
        b_dat11[i] = (MIOPEN_TYPE)0;
        b_dat12[i] = (MIOPEN_TYPE)0;
        b_dat13[i] = (MIOPEN_TYPE)0;
        b_dat14[i] = (MIOPEN_TYPE)0;
        b_dat15[i] = (MIOPEN_TYPE)0;
        b_dat16[i] = (MIOPEN_TYPE)0;
    }

    const int gid_        = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_size = gridDim.x * blockDim.x;
    for(int gid = gid_; gid < total_work; gid += global_size)
    {
        for(int i = 0; i < RD_BLCK; ++i)
        {
            a_dat[i] = (MIOPEN_TYPE)0;
            c_dat[i] = (MIOPEN_TYPE)0;
        }

        int io_index = gid * RD_BLCK;
        if(use_apl0 == 1)
        {
            *((READ_TYPE*)a_dat) = *((const READ_TYPE*)(a + Aoffset + io_index));
            for(int i = 0; i < RD_BLCK; ++i)
            {
                a_dat[i] *= alpha0;
            }
        }

        if(use_bet == 1)
        {
            *((READ_TYPE*)c_dat) = *((const READ_TYPE*)(c + Coffset + io_index));
            for(int i = 0; i < RD_BLCK; ++i)
            {
                c_dat[i] *= beta;
            }
        }

        g_RD_BLCK = gid * RD_BLCK;
        if(use_apl1 == 1)
        {
            for(int bid = 0; bid < ((b_c / 16) * 16); bid += 16)
            {

                int b_index1           = (bid * b_nstride) + g_RD_BLCK;
                int b_index2           = ((bid + 1) * b_nstride) + g_RD_BLCK;
                int b_index3           = ((bid + 2) * b_nstride) + g_RD_BLCK;
                int b_index4           = ((bid + 3) * b_nstride) + g_RD_BLCK;
                int b_index5           = ((bid + 4) * b_nstride) + g_RD_BLCK;
                int b_index6           = ((bid + 5) * b_nstride) + g_RD_BLCK;
                int b_index7           = ((bid + 6) * b_nstride) + g_RD_BLCK;
                int b_index8           = ((bid + 7) * b_nstride) + g_RD_BLCK;
                int b_index9           = ((bid + 8) * b_nstride) + g_RD_BLCK;
                int b_index10          = ((bid + 9) * b_nstride) + g_RD_BLCK;
                int b_index11          = ((bid + 10) * b_nstride) + g_RD_BLCK;
                int b_index12          = ((bid + 11) * b_nstride) + g_RD_BLCK;
                int b_index13          = ((bid + 12) * b_nstride) + g_RD_BLCK;
                int b_index14          = ((bid + 13) * b_nstride) + g_RD_BLCK;
                int b_index15          = ((bid + 14) * b_nstride) + g_RD_BLCK;
                int b_index16          = ((bid + 15) * b_nstride) + g_RD_BLCK;
                *((READ_TYPE*)b_dat1)  = *((const READ_TYPE*)(b + Boffset + b_index1));
                *((READ_TYPE*)b_dat2)  = *((const READ_TYPE*)(b + Boffset + b_index2));
                *((READ_TYPE*)b_dat3)  = *((const READ_TYPE*)(b + Boffset + b_index3));
                *((READ_TYPE*)b_dat4)  = *((const READ_TYPE*)(b + Boffset + b_index4));
                *((READ_TYPE*)b_dat5)  = *((const READ_TYPE*)(b + Boffset + b_index5));
                *((READ_TYPE*)b_dat6)  = *((const READ_TYPE*)(b + Boffset + b_index6));
                *((READ_TYPE*)b_dat7)  = *((const READ_TYPE*)(b + Boffset + b_index7));
                *((READ_TYPE*)b_dat8)  = *((const READ_TYPE*)(b + Boffset + b_index8));
                *((READ_TYPE*)b_dat9)  = *((const READ_TYPE*)(b + Boffset + b_index9));
                *((READ_TYPE*)b_dat10) = *((const READ_TYPE*)(b + Boffset + b_index10));
                *((READ_TYPE*)b_dat11) = *((const READ_TYPE*)(b + Boffset + b_index11));
                *((READ_TYPE*)b_dat12) = *((const READ_TYPE*)(b + Boffset + b_index12));
                *((READ_TYPE*)b_dat13) = *((const READ_TYPE*)(b + Boffset + b_index13));
                *((READ_TYPE*)b_dat14) = *((const READ_TYPE*)(b + Boffset + b_index14));
                *((READ_TYPE*)b_dat15) = *((const READ_TYPE*)(b + Boffset + b_index15));
                *((READ_TYPE*)b_dat16) = *((const READ_TYPE*)(b + Boffset + b_index16));

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat1[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat2[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat3[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat4[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat5[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat6[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat7[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat8[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat9[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat10[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat11[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat12[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat13[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat14[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat15[i] * alpha1);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat16[i] * alpha1);
                }
            }
            for(int bid = ((b_c / 16) * 16); bid < b_c; bid++)
            {
                int b_index           = bid * b_nstride + g_RD_BLCK;
                *((READ_TYPE*)b_dat1) = *((const READ_TYPE*)(b + Boffset + b_index));

                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], b_dat1[i] * alpha1);
                }
            }
        }
        else
        {
            for(int bid = 0; bid < ((b_c / 16) * 16); bid += 16)
            {
                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                }
            }
            for(int bid = ((b_c / 16) * 16); bid < b_c; bid++)
            {
                for(int i = 0; i < RD_BLCK; ++i)
                {
                    c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i], (MIOPEN_TYPE)0);
                }
            }
        }
        *((READ_TYPE*)(c + Coffset + io_index)) = *((READ_TYPE*)c_dat);
    }
}
#endif

#ifdef USE_3D_TENSOR_GENERIC
// NCH
extern "C" __global__ void Op3dTensorGeneric(const MIOPEN_TYPE* a,
                                             const MIOPEN_TYPE* b,
                                             MIOPEN_TYPE* c,
                                             const uint64_t Aoffset,
                                             const uint64_t Boffset,
                                             const uint64_t Coffset,
                                             const uint32_t b_c,
                                             const uint32_t b_h,
                                             const uint32_t c_c,
                                             const uint32_t c_h,
                                             const uint32_t a_nstride,
                                             const uint32_t a_cstride,
                                             const uint32_t a_hstride,
                                             const uint32_t b_nstride,
                                             const uint32_t b_cstride,
                                             const uint32_t b_hstride,
                                             const uint32_t c_nstride,
                                             const uint32_t c_cstride,
                                             const uint32_t c_hstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const uint32_t total_work,
                                             const bool use_beta)
{
    const MIOPEN_TYPE* a_off = a + Aoffset;
    const MIOPEN_TYPE* b_off = b + Boffset;
    MIOPEN_TYPE* c_off       = c + Coffset;

    auto gid          = blockIdx.x * blockDim.x + threadIdx.x;
    const auto* a_ptr = a_off + (gid / (c_c * c_h)) * a_nstride +
                        ((gid % (c_c * c_h)) / c_h) * a_cstride +
                        ((gid % (c_c * c_h)) % c_h) * a_hstride;
    auto* c_ptr = c_off + (gid / (c_c * c_h)) * c_nstride +
                  ((gid % (c_c * c_h)) / c_h) * c_cstride + ((gid % (c_c * c_h)) % c_h) * c_hstride;

    const auto step   = gridDim.x * blockDim.x;
    const auto a_step = (step / (c_c * c_h)) * a_nstride +
                        ((step % (c_c * c_h)) / c_h) * a_cstride +
                        ((step % (c_c * c_h)) % c_h) * a_hstride;

    const auto c_step = (step / (c_c * c_h)) * c_nstride +
                        ((step % (c_c * c_h)) / c_h) * c_cstride +
                        ((step % (c_c * c_h)) % c_h) * c_hstride;

    const auto c_end = c_off + total_work * c_nstride;
    while(c_ptr < c_end)
    {
        const auto* b_ptr = b_off;
        if(b_nstride != 0)
            b_ptr += (gid / (b_c * b_h)) * b_nstride;

        if(b_cstride != 0)
            b_ptr += ((gid % (b_c * b_h)) / b_h) * b_cstride;

        if(b_hstride != 0)
            b_ptr += ((gid % (b_c * b_h)) % b_h) * b_hstride;

        auto b_val = *b_ptr;
        auto a_val = *a_ptr;
        auto c_val = use_beta ? *c_ptr : static_cast<MIOPEN_TYPE>(0);
        *c_ptr     = MIOPEN_TENSOR_OP(b_val * alpha1, a_val * alpha0) + c_val * beta;

        a_ptr += a_step;
        c_ptr += c_step;
        gid += step;
    }
}

#endif

#ifdef USE_4D_TENSOR_GENERIC
// NCHW
extern "C" __global__ void Op4dTensorGeneric(MIOPEN_TYPE* a,
                                             const int a_nstride,
                                             const int a_cstride,
                                             const int a_hstride,
                                             MIOPEN_TYPE* b,
                                             const int b_c,
                                             const int b_h,
                                             const int b_w,
                                             const int b_nstride,
                                             const int b_cstride,
                                             const int b_hstride,
                                             MIOPEN_TYPE* c,
                                             const int c_c,
                                             const int c_h,
                                             const int c_w,
                                             const int c_nstride,
                                             const int c_cstride,
                                             const int c_hstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const unsigned int bitmap,
                                             const int work_per_wg,
                                             const long Aoffset,
                                             const long Boffset,
                                             const long Coffset,
                                             const int num_wg)
{
    int gid = blockIdx.x;

    MIOPEN_TYPE* a_off = a + Aoffset;
    MIOPEN_TYPE* b_off = b + Boffset;
    MIOPEN_TYPE* c_off = c + Coffset;

// MIOPEN_TYPE operand = b[gid + Boffset];
// num_wg: the number of workgroups should be launched
// MAX_NUM_WG: the maximum number of workgroups actually launched
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
    if(beta == static_cast<MIOPEN_TYPE>(0))
#pragma clang diagnostic pop
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid = threadIdx.x;

            int o_h_div = (bitmap & (1 << 0)) ? 1 : c_w;
            int o_c_div = o_h_div * ((bitmap & (1 << 1)) ? 1 : c_h);
            int o_n_div = o_c_div * ((bitmap & (1 << 2)) ? 1 : c_c);

            int o_w_gid_off = gid % b_w;
            int o_h_gid_off = (gid / b_w) % b_h;
            int o_c_gid_off = (gid / b_w / b_h) % b_c;
            int o_n_gid_off = (gid / b_w / b_h) / b_c;

            int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride +
                         o_h_gid_off * b_hstride + o_w_gid_off;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_w = (bitmap & (1 << 0)) ? o_w_gid_off : lid % c_w;
                int o_h = (bitmap & (1 << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
                int o_c = (bitmap & (1 << 2)) ? o_c_gid_off : (lid / o_c_div) % c_c;
                int o_n = (bitmap & (1 << 3)) ? o_n_gid_off : lid / o_n_div;

                int aindex    = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex    = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] = MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand);

                lid += blockDim.x;
            }
        }
    }
    else
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid = threadIdx.x;

            int o_h_div = (bitmap & (1 << 0)) ? 1 : c_w;
            int o_c_div = o_h_div * ((bitmap & (1 << 1)) ? 1 : c_h);
            int o_n_div = o_c_div * ((bitmap & (1 << 2)) ? 1 : c_c);

            int o_w_gid_off = gid % b_w;
            int o_h_gid_off = (gid / b_w) % b_h;
            int o_c_gid_off = (gid / b_w / b_h) % b_c;
            int o_n_gid_off = (gid / b_w / b_h) / b_c;

            int bindex = o_n_gid_off * b_nstride + o_c_gid_off * b_cstride +
                         o_h_gid_off * b_hstride + o_w_gid_off;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_w = (bitmap & (1 << 0)) ? o_w_gid_off : lid % c_w;
                int o_h = (bitmap & (1 << 1)) ? o_h_gid_off : (lid / o_h_div) % c_h;
                int o_c = (bitmap & (1 << 2)) ? o_c_gid_off : (lid / o_c_div) % c_c;
                int o_n = (bitmap & (1 << 3)) ? o_n_gid_off : lid / o_n_div;

                int aindex = o_n * a_nstride + o_c * a_cstride + o_h * a_hstride + o_w;
                int cindex = o_n * c_nstride + o_c * c_cstride + o_h * c_hstride + o_w;
                c_off[cindex] =
                    MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];

                lid += blockDim.x;
            }
        }
    }
}

#endif

#ifdef USE_4D_TENSOR_LITE
extern "C" __global__ void Op4dTensorLite(const MIOPEN_TYPE* a,
                                          const MIOPEN_TYPE* b,
                                          MIOPEN_TYPE* c,
                                          const MIOPEN_TYPE alpha0,
                                          const MIOPEN_TYPE alpha1,
                                          const MIOPEN_TYPE beta,
                                          const long Aoffset,
                                          const long Boffset,
                                          const long Coffset,
                                          const long total_work,
                                          const int use_beta)
{
    int gid0        = blockIdx.x * blockDim.x + threadIdx.x;
    int global_size = gridDim.x * blockDim.x;

    MIOPEN_TYPE a_dat[RD_BLCK];
    MIOPEN_TYPE b_dat[RD_BLCK];
    MIOPEN_TYPE c_dat[RD_BLCK];

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
    if(beta == static_cast<MIOPEN_TYPE>(0))
#pragma clang diagnostic pop
    {
        for(; gid0 < total_work; gid0 += global_size)
        {
            int index = gid0 * RD_BLCK;

            for(int i = 0; i < RD_BLCK; ++i)
            {
                c_dat[i] = static_cast<MIOPEN_TYPE>(0);
            }

            *(reinterpret_cast<READ_TYPE*>(a_dat)) =
                *(reinterpret_cast<const READ_TYPE*>(a + index + Aoffset));
            *(reinterpret_cast<READ_TYPE*>(b_dat)) =
                *(reinterpret_cast<const READ_TYPE*>(b + index + Boffset));
            if(use_beta == 1)
            {
                *(reinterpret_cast<READ_TYPE*>(c_dat)) =
                    *(reinterpret_cast<const READ_TYPE*>(c + index + Coffset));
            }

            for(int i = 0; i < RD_BLCK; ++i)
            {
                if(use_beta == 1)
                {
                    c_dat[i] = static_cast<MIOPEN_TYPE>(0);
                }
                c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i] * alpha0, b_dat[i] * alpha1);
            }

            *(reinterpret_cast<READ_TYPE*>(c + index + Coffset)) =
                *(reinterpret_cast<READ_TYPE*>(c_dat));
        }
    }
    else
    {
        for(; gid0 < total_work; gid0 += global_size)
        {
            int index = gid0 * RD_BLCK;

            for(int i = 0; i < RD_BLCK; ++i)
            {
                c_dat[i] = (MIOPEN_TYPE)0;
            }

            *(reinterpret_cast<READ_TYPE*>(a_dat)) =
                *(reinterpret_cast<const READ_TYPE*>(a + index + Aoffset));
            *(reinterpret_cast<READ_TYPE*>(b_dat)) =
                *(reinterpret_cast<const READ_TYPE*>(b + index + Boffset));
            if(use_beta == 1)
            {
                *(reinterpret_cast<READ_TYPE*>(c_dat)) =
                    *(reinterpret_cast<const READ_TYPE*>(c + index + Coffset));
            }

            for(int i = 0; i < RD_BLCK; ++i)
            {
                if(use_beta == 1)
                {
                    c_dat[i] *= beta;
                }
                c_dat[i] += MIOPEN_TENSOR_OP(a_dat[i] * alpha0, b_dat[i] * alpha1);
            }

            *(reinterpret_cast<READ_TYPE*>(c + index + Coffset)) =
                *(reinterpret_cast<READ_TYPE*>(c_dat));
        }
    }
}
#endif // USE_4D_TENSOR_LITE