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
                                             const uint32_t a_nstride,
                                             const uint32_t b_nstride,
                                             const uint32_t c_nstride,
                                             const MIOPEN_TYPE alpha0,
                                             const MIOPEN_TYPE alpha1,
                                             const MIOPEN_TYPE beta,
                                             const uint32_t total_work,
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
extern "C" __global__ void Op2dTensorGeneric(MIOPEN_TYPE* a,
                                             const int a_nstride,
                                             MIOPEN_TYPE* b,
                                             const int b_c,
                                             const int b_nstride,
                                             MIOPEN_TYPE* c,
                                             const int c_c,
                                             const int c_nstride,
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

    int o_n_div = (bitmap & (1 << 0)) ? 1 : c_c;

    // num_wg: the number of workgroups should be launched
    // MAX_NUM_WG: the maximum number of workgroups actually launched
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
    if(beta == static_cast<MIOPEN_TYPE>(0))
#pragma clang diagnostic pop
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {

            int lid         = threadIdx.x;
            int o_c_gid_off = gid % b_c;
            int o_n_gid_off = gid / b_c;

            int bindex          = o_n_gid_off * b_nstride + o_c_gid_off;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_c       = (bitmap & (1 << 0)) ? o_c_gid_off : lid % c_c;
                int o_n       = (bitmap & (1 << 1)) ? o_n_gid_off : lid / o_n_div;
                int aindex    = o_n * a_nstride + o_c;
                int cindex    = o_n * c_nstride + o_c;
                c_off[cindex] = MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand);
                lid += blockDim.x;
            }
        }
    }
    else
    {
        for(; gid < num_wg; gid += MAX_NUM_WG)
        {
            int lid         = threadIdx.x;
            int o_c_gid_off = gid % b_c;
            int o_n_gid_off = gid / b_c;

            int bindex          = o_n_gid_off * b_nstride + o_c_gid_off;
            MIOPEN_TYPE operand = b_off[bindex] * alpha1;

            while(lid < work_per_wg)
            {
                int o_c    = (bitmap & (1 << 0)) ? o_c_gid_off : lid % c_c;
                int o_n    = (bitmap & (1 << 1)) ? o_n_gid_off : lid / o_n_div;
                int aindex = o_n * a_nstride + o_c;
                int cindex = o_n * c_nstride + o_c;
                c_off[cindex] =
                    MIOPEN_TENSOR_OP(a_off[aindex] * alpha0, operand) + beta * c_off[cindex];
                lid += blockDim.x;
            }
        }
    }
}

#endif