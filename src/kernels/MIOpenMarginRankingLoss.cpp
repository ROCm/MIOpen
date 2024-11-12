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

#include "float_types.h"
#include "tensor_view.hpp"

template <typename DT, int REDUCTION_T>
__device__ void marginRankingLossForward5d(const DT* __restrict__ I1,
                                           const DT* __restrict__ I2,
                                           const DT* __restrict__ T,
                                           void* __restrict__ O,
                                           float margin,
                                           float divisor,
                                           tensor_view_t<5> I1_tv,
                                           tensor_view_t<5> I2_tv,
                                           tensor_view_t<5> T_tv,
                                           tensor_view_t<5> O_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t n0123 = gid / I1_tv.size[4], n4 = gid % I1_tv.size[4];
    uint64_t n012 = n0123 / I1_tv.size[3], n3 = n0123 % I1_tv.size[3];
    uint64_t n01 = n012 / I1_tv.size[2], n2 = n012 % I1_tv.size[2];
    uint64_t n0 = n01 / I1_tv.size[1], n1 = n01 % I1_tv.size[1];

    if(!(n0 < I1_tv.size[0]))
        return;

    uint64_t I1idx = I1_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
    uint64_t I2idx = I2_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
    uint64_t Tidx  = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

    FLOAT_ACCUM output_accum =
        -CVT_FLOAT2ACCUM(T[Tidx]) * (CVT_FLOAT2ACCUM(I1[I1idx]) - CVT_FLOAT2ACCUM(I2[I2idx])) +
        static_cast<FLOAT_ACCUM>(margin);
    if(output_accum < 0.0f)
        output_accum = 0.0f;

    switch(REDUCTION_T)
    {
    case 0: {
        uint64_t Oidx             = O_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        static_cast<DT*>(O)[Oidx] = CVT_ACCUM2FLOAT(output_accum);
        break;
    }
    case 1: {
        static_cast<FLOAT_ACCUM*>(O)[gid] = output_accum;
        break;
    }
    case 2: {
        static_cast<FLOAT_ACCUM*>(O)[gid] = output_accum / static_cast<FLOAT_ACCUM>(divisor);
        break;
    }
    default: break;
    }
}

extern "C" __global__ void MarginRankingLossForward5d(const DTYPE* __restrict__ I1,
                                                      const DTYPE* __restrict__ I2,
                                                      const DTYPE* __restrict__ T,
                                                      DTYPE* __restrict__ O,
                                                      float margin,
                                                      float divisor,
                                                      tensor_view_t<5> I1_tv,
                                                      tensor_view_t<5> I2_tv,
                                                      tensor_view_t<5> T_tv,
                                                      tensor_view_t<5> O_tv)
{
    marginRankingLossForward5d<DTYPE, REDUCTION_TYPE>(
        I1, I2, T, O, margin, divisor, I1_tv, I2_tv, T_tv, O_tv);
}

template <typename DT, int REDUCTION_T>
__device__ void marginRankingLossBackward5d(const DT* __restrict__ I1,
                                            const DT* __restrict__ I2,
                                            const DT* __restrict__ T,
                                            const DT* __restrict__ dO,
                                            DT* __restrict__ dI1,
                                            DT* __restrict__ dI2,
                                            float margin,
                                            float divisor,
                                            tensor_view_t<5> I1_tv,
                                            tensor_view_t<5> I2_tv,
                                            tensor_view_t<5> T_tv,
                                            tensor_view_t<5> dO_tv,
                                            tensor_view_t<5> dI1_tv,
                                            tensor_view_t<5> dI2_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t n0123 = gid / I1_tv.size[4], n4 = gid % I1_tv.size[4];
    uint64_t n012 = n0123 / I1_tv.size[3], n3 = n0123 % I1_tv.size[3];
    uint64_t n01 = n012 / I1_tv.size[2], n2 = n012 % I1_tv.size[2];
    uint64_t n0 = n01 / I1_tv.size[1], n1 = n01 % I1_tv.size[1];

    if(!(n0 < I1_tv.size[0]))
        return;

    uint64_t I1idx  = I1_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
    uint64_t I2idx  = I2_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
    uint64_t dI1idx = dI1_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
    uint64_t dI2idx = dI2_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
    uint64_t Tidx   = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

    FLOAT_ACCUM t =
        -CVT_FLOAT2ACCUM(T[Tidx]) * (CVT_FLOAT2ACCUM(I1[I1idx]) - CVT_FLOAT2ACCUM(I2[I2idx])) +
        static_cast<FLOAT_ACCUM>(margin);

    if(t < 0)
    {
        dI1[dI1idx] = 0.0f;
        dI2[dI2idx] = 0.0f;
    }
    else
    {
        FLOAT_ACCUM d_accum;
        switch(REDUCTION_T)
        {
        case 0: {
            uint64_t dOidx = dO_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
            d_accum        = CVT_FLOAT2ACCUM(T[Tidx]) * CVT_FLOAT2ACCUM(dO[dOidx]);
            break;
        }
        case 1: {
            d_accum = CVT_FLOAT2ACCUM(T[Tidx]) * CVT_FLOAT2ACCUM(dO[0]);
            break;
        }
        case 2: {
            d_accum = CVT_FLOAT2ACCUM(T[Tidx]) * CVT_FLOAT2ACCUM(dO[0]) /
                      static_cast<FLOAT_ACCUM>(divisor);
            break;
        }
        default: break;
        }
        dI1[dI1idx] = CVT_ACCUM2FLOAT(-d_accum);
        dI2[dI2idx] = CVT_ACCUM2FLOAT(d_accum);
    }
}

extern "C" __global__ void MarginRankingLossBackward5d(const DTYPE* __restrict__ I1,
                                                       const DTYPE* __restrict__ I2,
                                                       const DTYPE* __restrict__ T,
                                                       const DTYPE* __restrict__ dO,
                                                       DTYPE* __restrict__ dI1,
                                                       DTYPE* __restrict__ dI2,
                                                       float margin,
                                                       float divisor,
                                                       tensor_view_t<5> I1_tv,
                                                       tensor_view_t<5> I2_tv,
                                                       tensor_view_t<5> T_tv,
                                                       tensor_view_t<5> dO_tv,
                                                       tensor_view_t<5> dI1_tv,
                                                       tensor_view_t<5> dI2_tv)
{
    marginRankingLossBackward5d<DTYPE, REDUCTION_TYPE>(
        I1, I2, T, dO, dI1, dI2, margin, divisor, I1_tv, I2_tv, T_tv, dO_tv, dI1_tv, dI2_tv);
}
