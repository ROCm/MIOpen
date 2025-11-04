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

#ifdef USE_5D_TENSOR_GENERIC

// register-level packet processing to improve parallelism and reduce div/mod overhead.
// PACK_T = how many consecutive elements each thread handles per inner iteration.
// If the solver didn't define it at JIT time, fall back to 8 for standalone builds.
#ifndef PACK_T
#define PACK_T 8
#endif

static_assert(PACK_T >= 1 && PACK_T <= 16, "PACK_T must be in [1..16]");

// Adaptive selection between the narrow (32-bit) and wide (64-bit) index/offset types.
// 32-bit for faster arithmetic when the address range allows that. 64-bit
// to safely handle very large tensors or strides/offsets.
#ifdef USE_INDEX32
    using offset_t = uint64_t;
    using len_t    = unsigned int;
    using idx_t    = uint32_t;
#else
    using offset_t = uint64_t;
    using len_t    = unsigned long long;
    using idx_t    = uint64_t;
#endif

// below - __restrict__ is used to allow compiler to aggressively optimize load/read operations
// on tensors since we guarantee that there is no overlap between a, b, c pointers
extern "C" __global__ void Op5dTensorGeneric(const MIOPEN_TYPE* __restrict__ a,
                                             const MIOPEN_TYPE* __restrict__ b,
                                             MIOPEN_TYPE* __restrict__ c,
                                             const offset_t Aoffset,
                                             const offset_t Boffset,
                                             const offset_t Coffset,
                                             const len_t b_n,
                                             const len_t b_c,
                                             const len_t b_d,
                                             const len_t b_h,
                                             const len_t b_w,
                                             const len_t c_n,
                                             const len_t c_c,
                                             const len_t c_d,
                                             const len_t c_h,
                                             const len_t c_w,
                                             const len_t a_nstride,
                                             const len_t a_cstride,
                                             const len_t a_dstride,
                                             const len_t a_hstride,
                                             const len_t a_wstride,
                                             const len_t b_nstride,
                                             const len_t b_cstride,
                                             const len_t b_dstride,
                                             const len_t b_hstride,
                                             const len_t b_wstride,
                                             const len_t c_nstride,
                                             const len_t c_cstride,
                                             const len_t c_dstride,
                                             const len_t c_hstride,
                                             const len_t c_wstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const len_t total_work,
                                             const bool use_beta)
{
    const MIOPEN_TYPE* a_base = a + static_cast<size_t>(Aoffset);
    const MIOPEN_TYPE* b_base = b + static_cast<size_t>(Boffset);
    MIOPEN_TYPE* c_base       = c + static_cast<size_t>(Coffset);

// USE_PACKED_INNER toggles this optimization on
#ifndef USE_PACKED_INNER
#define USE_PACKED_INNER 1
#endif

    // calc thread index and total thread count
    const idx_t tcount = static_cast<idx_t>(blockDim.x) * static_cast<idx_t>(gridDim.x);
    const idx_t tid    = static_cast<idx_t>(blockIdx.x) * static_cast<idx_t>(blockDim.x)
                      + static_cast<idx_t>(threadIdx.x);

#if USE_PACKED_INNER
    // keep wide_t strictly 64-bit to avoid overflow in index calculations
    using wide_t = uint64_t;

    // each thread processes multiple PACK_T-sized blocks, cast to wider type only for the
    // current packs amount and total work, to avoid potential overflow
    for (wide_t i = static_cast<wide_t>(tid); i * static_cast<wide_t>(PACK_T) < static_cast<wide_t>(total_work); i += static_cast<wide_t>(tcount)){

        // additional guard to avoid internal overflow when using 32-bit idx_t
        const wide_t base = i * static_cast<wide_t>(PACK_T);

        // Decompose indices for C ( == A) per one block of PACK_T elements
        const wide_t cw  = static_cast<wide_t>(c_w);
        const wide_t ch  = static_cast<wide_t>(c_h);
        const wide_t cd  = static_cast<wide_t>(c_d);
        const wide_t ccw = static_cast<wide_t>(c_c);

        wide_t tmp_w = base;

        const idx_t w0 = static_cast<idx_t>(tmp_w % cw);  tmp_w /= cw;
        const idx_t h0 = static_cast<idx_t>(tmp_w % ch);  tmp_w /= ch;
        const idx_t d0 = static_cast<idx_t>(tmp_w % cd);  tmp_w /= cd;
        const idx_t c0 = static_cast<idx_t>(tmp_w % ccw); tmp_w /= ccw;
        const idx_t n0 = static_cast<idx_t>(tmp_w);

        // Broadcast indices for B, each dimension of B may be either 1, i.e.
        // broadcast or equal to the corresponding C dim.
        const idx_t bn0 = (b_n == 1) ? 0 : ((b_n == c_n) ? n0 : (n0 % static_cast<idx_t>(b_n)));
        const idx_t bc0 = (b_c == 1) ? 0 : ((b_c == c_c) ? c0 : (c0 % static_cast<idx_t>(b_c)));
        const idx_t bd0 = (b_d == 1) ? 0 : ((b_d == c_d) ? d0 : (d0 % static_cast<idx_t>(b_d)));
        const idx_t bh0 = (b_h == 1) ? 0 : ((b_h == c_h) ? h0 : (h0 % static_cast<idx_t>(b_h)));

        // Check W for possible broadcasting for A and B When true,
        // the same element reused for all W positions for A only case with stride == 0
        // for B: stride == 0 or size == 1
        const bool aw0 = (a_wstride == 0);
        const bool bw0 = (b_w == 1) || (b_wstride == 0);

        // Base offsets for A, B, C tensors
        wide_t a_off = static_cast<wide_t>(n0) * a_nstride
                            + static_cast<wide_t>(c0) * a_cstride
                            + static_cast<wide_t>(d0) * a_dstride
                            + static_cast<wide_t>(h0) * a_hstride
                            + (aw0 ? static_cast<wide_t>(0) : static_cast<wide_t>(w0) * a_wstride);

        wide_t b_off = static_cast<wide_t>(bn0) * b_nstride
                     + static_cast<wide_t>(bc0) * b_cstride
                     + static_cast<wide_t>(bd0) * b_dstride
                     + static_cast<wide_t>(bh0) * b_hstride
                     + (bw0 ? static_cast<wide_t>(0) : static_cast<wide_t>(w0) * b_wstride);

        wide_t c_off = static_cast<wide_t>(n0) * c_nstride
                    + static_cast<wide_t>(c0) * c_cstride
                    + static_cast<wide_t>(d0) * c_dstride
                    + static_cast<wide_t>(h0) * c_hstride
                    + static_cast<wide_t>(w0) * c_wstride;

        // Increments for A, B, C ptrs
        const wide_t a_inc_w = aw0 ? static_cast<wide_t>(0) : static_cast<wide_t>(a_wstride);
        const wide_t b_inc_w = bw0 ? static_cast<wide_t>(0) : static_cast<wide_t>(b_wstride);
        const wide_t c_inc_w = static_cast<wide_t>(c_wstride);

        // Remaining elements in current C-row by W. Compute via wide_t then narrow.
        const wide_t w_rem_w = cw - static_cast<wide_t>(w0);
        const idx_t  w_run   = static_cast<idx_t>(w_rem_w < static_cast<wide_t>(PACK_T) ? w_rem_w : static_cast<wide_t>(PACK_T));

        // for broadcasted A or B, cache the reused value - save redundant loads
        MIOPEN_TYPE av_cached = MIOPEN_TYPE(0);
        if(aw0) av_cached = a_base[static_cast<size_t>(a_off)];

        MIOPEN_TYPE bv_cached = MIOPEN_TYPE(0);
        if(bw0) bv_cached = b_base[static_cast<size_t>(b_off)];

        // "unroll" below - to unwind small simple loops for better
        // performance through improving instruction-level parallelism
        // assumed for pack sizes 8 or 16, otherwise may reduce performance
        #pragma unroll PACK_T

        // execute operation for the main part of the pack
        for (idx_t k = 0; k < static_cast<idx_t>(PACK_T); ++k) {
            if (k < w_run) {
                const MIOPEN_TYPE av  = aw0 ? av_cached : a_base[static_cast<size_t>(a_off)];
                const MIOPEN_TYPE bv  = bw0 ? bv_cached : b_base[static_cast<size_t>(b_off)];
                const MIOPEN_TYPE res = MIOPEN_TENSOR_OP(av * alpha0, bv * alpha1);

                if (!use_beta) {
                    c_base[static_cast<size_t>(c_off)] = res;
                }
                else {
                    const MIOPEN_TYPE cv = c_base[static_cast<size_t>(c_off)];
                    c_base[static_cast<size_t>(c_off)] = res + cv * beta;
                }

                a_off += a_inc_w;
                b_off += b_inc_w;
                c_off += c_inc_w;
            }
        }

        // process remaining elements (tail) if any are present
        for (idx_t k = w_run; k < static_cast<idx_t>(PACK_T); ++k)
        {
            const wide_t idx_w = base + static_cast<wide_t>(k);
            if (idx_w >= static_cast<wide_t>(total_work)) break;

            // decompose linear index using cw/ch/cd/ccw defined above
            const idx_t w  = static_cast<idx_t>( idx_w % cw );
            const idx_t h  = static_cast<idx_t>((idx_w / cw) % ch);
            const idx_t d  = static_cast<idx_t>((idx_w / (cw * ch)) % cd);
            const idx_t c1 = static_cast<idx_t>((idx_w / (cw * ch * cd)) % ccw);
            const idx_t n  = static_cast<idx_t>( idx_w / (cw * ch * cd * ccw) );

            // B broadcast mapping
            const idx_t bn = (b_n == 1) ? 0 : ((b_n == c_n) ? n  : (n  % static_cast<idx_t>(b_n)));
            const idx_t bc = (b_c == 1) ? 0 : ((b_c == c_c) ? c1 : (c1 % static_cast<idx_t>(b_c)));
            const idx_t bd = (b_d == 1) ? 0 : ((b_d == c_d) ? d  : (d  % static_cast<idx_t>(b_d)));
            const idx_t bh = (b_h == 1) ? 0 : ((b_h == c_h) ? h  : (h  % static_cast<idx_t>(b_h)));
            const idx_t bw = (b_w == 1) ? 0 : ((b_w == c_w) ? w  : (w  % static_cast<idx_t>(b_w)));

            // offsets
            const wide_t a_off_tail = static_cast<wide_t>(n)  * a_nstride
                                    + static_cast<wide_t>(c1) * a_cstride
                                    + static_cast<wide_t>(d)  * a_dstride
                                    + static_cast<wide_t>(h)  * a_hstride
                                    + static_cast<wide_t>(w)  * a_wstride;

            const wide_t b_off_tail = static_cast<wide_t>(bn) * b_nstride
                                    + static_cast<wide_t>(bc) * b_cstride
                                    + static_cast<wide_t>(bd) * b_dstride
                                    + static_cast<wide_t>(bh) * b_hstride
                                    + static_cast<wide_t>(bw) * b_wstride;

            const wide_t c_off_tail = static_cast<wide_t>(n)  * c_nstride
                                    + static_cast<wide_t>(c1) * c_cstride
                                    + static_cast<wide_t>(d)  * c_dstride
                                    + static_cast<wide_t>(h)  * c_hstride
                                    + static_cast<wide_t>(w)  * c_wstride;

            // compute
            const MIOPEN_TYPE av  = a_base[static_cast<size_t>(a_off_tail)];
            const MIOPEN_TYPE bv_tail = bw0 ? bv_cached : b_base[static_cast<size_t>(b_off_tail)];
            const MIOPEN_TYPE tmp = MIOPEN_TENSOR_OP(av * alpha0, bv_tail * alpha1);

            if (!use_beta) {
                c_base[static_cast<size_t>(c_off_tail)] = tmp;
            } else {
                const MIOPEN_TYPE cv = c_base[static_cast<size_t>(c_off_tail)];
                c_base[static_cast<size_t>(c_off_tail)] = tmp + cv * beta;
            }
        }
    }
#else
    // scalar version without inner packing
    using wide_t = uint64_t;

    #pragma unroll 1
    for (wide_t i = static_cast<wide_t>(tid);
         i < static_cast<wide_t>(total_work);
         i += static_cast<wide_t>(tcount))
    {
        // widen dims once
        const wide_t cw  = static_cast<wide_t>(c_w);
        const wide_t ch  = static_cast<wide_t>(c_h);
        const wide_t cd  = static_cast<wide_t>(c_d);
        const wide_t ccw = static_cast<wide_t>(c_c);

        // decompose linear index (в wide_t), затем сужаем в idx_t
        const idx_t w  = static_cast<idx_t>( i % cw );
        const idx_t h  = static_cast<idx_t>((i / cw) % ch);
        const idx_t d  = static_cast<idx_t>((i / (cw * ch)) % cd);
        const idx_t c1 = static_cast<idx_t>((i / (cw * ch * cd)) % ccw);
        const idx_t n  = static_cast<idx_t>(  i / (cw * ch * cd * ccw) );

        // B broadcast mapping (len_t -> idx_t)
        const idx_t bn = (b_n == 1) ? 0 : ((b_n == c_n) ? n  : (n  % static_cast<idx_t>(b_n)));
        const idx_t bc = (b_c == 1) ? 0 : ((b_c == c_c) ? c1 : (c1 % static_cast<idx_t>(b_c)));
        const idx_t bd = (b_d == 1) ? 0 : ((b_d == c_d) ? d  : (d  % static_cast<idx_t>(b_d)));
        const idx_t bh = (b_h == 1) ? 0 : ((b_h == c_h) ? h  : (h  % static_cast<idx_t>(b_h)));
        const idx_t bw = (b_w == 1) ? 0 : ((b_w == c_w) ? w  : (w  % static_cast<idx_t>(b_w)));

        // offsets в idx_t
        const wide_t a_off =
            static_cast<wide_t>(n)  * a_nstride + static_cast<wide_t>(c1) * a_cstride +
            static_cast<wide_t>(d)  * a_dstride + static_cast<wide_t>(h)  * a_hstride +
            static_cast<wide_t>(w)  * a_wstride;

        const wide_t b_off =
            static_cast<wide_t>(bn) * b_nstride + static_cast<wide_t>(bc) * b_cstride +
            static_cast<wide_t>(bd) * b_dstride + static_cast<wide_t>(bh) * b_hstride +
            static_cast<wide_t>(bw) * b_wstride;

        const wide_t c_off =
            static_cast<wide_t>(n)  * c_nstride + static_cast<wide_t>(c1) * c_cstride +
            static_cast<wide_t>(d)  * c_dstride + static_cast<wide_t>(h)  * c_hstride +
            static_cast<wide_t>(w)  * c_wstride;

        // compute + store (избегаем лишнего чтения c_base при !use_beta)
        const MIOPEN_TYPE av  = a_base[static_cast<size_t>(a_off)];
        const MIOPEN_TYPE bv  = b_base[static_cast<size_t>(b_off)];
        const MIOPEN_TYPE tmp = MIOPEN_TENSOR_OP(av * alpha0, bv * alpha1);

        if (!use_beta) {
            c_base[static_cast<size_t>(c_off)] = tmp;
        } else {
            const MIOPEN_TYPE cv = c_base[static_cast<size_t>(c_off)];
            c_base[static_cast<size_t>(c_off)] = tmp + cv * beta;
        }
    }
#endif
}

extern "C" __global__ void Op5dTensorGenericContiguous(const MIOPEN_TYPE* __restrict__ a,
                                                       const MIOPEN_TYPE* __restrict__ b,
                                                       MIOPEN_TYPE* __restrict__ c,
                                                       const offset_t Aoffset,
                                                       const offset_t Boffset,
                                                       const offset_t Coffset,
                                                       [[maybe_unused]] const len_t b_n,
                                                       [[maybe_unused]] const len_t b_c,
                                                       [[maybe_unused]] const len_t b_d,
                                                       [[maybe_unused]] const len_t b_h,
                                                       const len_t b_w,
                                                       [[maybe_unused]] const len_t c_n,
                                                       [[maybe_unused]] const len_t c_c,
                                                       [[maybe_unused]] const len_t c_d,
                                                       [[maybe_unused]] const len_t c_h,
                                                       const len_t c_w,
                                                       const MIOPEN_TYPE alpha0,
                                                       const MIOPEN_TYPE alpha1,
                                                       const MIOPEN_TYPE beta,
                                                       const len_t total_work,
                                                       const bool use_beta)
{
    const MIOPEN_TYPE* a_base = a + static_cast<size_t>(Aoffset);
    const MIOPEN_TYPE* b_base = b + static_cast<size_t>(Boffset);
    MIOPEN_TYPE* c_base = c + static_cast<size_t>(Coffset);

    const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(blockDim.x) * gridDim.x;

    constexpr int pack_size = PACK_T;
    const int effective_pack = (c_w < 16) ? 1 : pack_size;

    for(uint64_t base = tid * effective_pack; base < total_work; base += stride * effective_pack)
    {
        // Process pack_size elements at a time
        MIOPEN_TYPE a_val[pack_size];
        MIOPEN_TYPE b_val[pack_size];
        MIOPEN_TYPE c_val[pack_size];

        const uint64_t remaining = (base + effective_pack <= total_work)
            ? effective_pack
            : total_work - base;

        // Load A and B values
        #pragma unroll
        for(int i = 0; i < pack_size; i++) {
            if(i < remaining) {
                a_val[i] = a_base[static_cast<size_t>(base) + i];
                b_val[i] = b_base[static_cast<size_t>(base) + i];
            }
        }

        // Load C if needed
        if(use_beta) {
            #pragma unroll
            for(int i = 0; i < pack_size; i++) {
                if(i < remaining) {
                    c_val[i] = c_base[static_cast<size_t>(base) + i];
                }
            }
        }

        // Compute
        #pragma unroll
        for(int i = 0; i < pack_size; i++) {
            if(i < remaining) {
                MIOPEN_TYPE tmp = MIOPEN_TENSOR_OP(a_val[i] * alpha0, b_val[i] * alpha1);
                c_val[i] = use_beta ? (tmp + beta * c_val[i]) : tmp;
            }
        }

        // Store results
        #pragma unroll
        for(int i = 0; i < pack_size; i++) {
            if(i < remaining) {
                c_base[static_cast<size_t>(base) + i] = c_val[i];
            }
        }
    }
}
#endif
