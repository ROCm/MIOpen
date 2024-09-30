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

template <typename IO_TYPE>
__device__ void DeviceMSELossForward5d(const IO_TYPE* __restrict__ I,
                                       const IO_TYPE* __restrict__ T,
                                       FLOAT_ACCUM* __restrict__ lsum,
                                       float divisor,
                                       tensor_view_t<5> I_tv,
                                       tensor_view_t<5> T_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
    size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
    size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
    size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

    if(!(n0 < I_tv.size[0]))
        return;

    size_t Iidx = I_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
    size_t Tidx = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

    FLOAT_ACCUM iidxval = CVT_FLOAT2ACCUM(I[Iidx]);
    FLOAT_ACCUM tidxval = CVT_FLOAT2ACCUM(T[Tidx]);
    FLOAT_ACCUM lsumval = (iidxval - tidxval) * (iidxval - tidxval) / divisor;

    lsum[gid] = lsumval;
}

template <typename IO_TYPE>
__device__ void DeviceMSELossBackward5d(const IO_TYPE* __restrict__ I,
                                        const IO_TYPE* __restrict__ T,
                                        const IO_TYPE* __restrict__ dO,
                                        IO_TYPE* __restrict__ dI,
                                        IO_TYPE* __restrict__ dT,
                                        float divisor,
                                        tensor_view_t<5> I_tv,
                                        tensor_view_t<5> T_tv,
                                        tensor_view_t<5> dO_tv,
                                        tensor_view_t<5> dI_tv,
                                        tensor_view_t<5> dT_tv)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
    size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
    size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
    size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

    if(!(n0 < I_tv.size[0]))
        return;

    size_t Iidx = I_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
    size_t Tidx = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

    FLOAT_ACCUM iidxval  = CVT_FLOAT2ACCUM(I[Iidx]);
    FLOAT_ACCUM tidxval  = CVT_FLOAT2ACCUM(T[Tidx]);
    FLOAT_ACCUM dOidxval = CVT_FLOAT2ACCUM(dO[dO_tv.get_tensor_view_idx({dO_tv, 0})]);
    FLOAT_ACCUM grad     = 2.0f * (iidxval - tidxval) / divisor * dOidxval;

    if(dI != nullptr)
    {
        size_t dIidx = dI_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        dI[dIidx]    = CVT_ACCUM2FLOAT(grad);
    }
    if(dT != nullptr)
    {
        size_t dTidx = dT_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        dT[dTidx]    = CVT_ACCUM2FLOAT(-grad);
    }
}
// Trampolines
extern "C" __global__ void MSELossForward5d(const FLOAT* __restrict__ I,
                                            const FLOAT* __restrict__ T,
                                            FLOAT_ACCUM* __restrict__ lsum,
                                            float divisor,
                                            tensor_view_t<5> I_tv,
                                            tensor_view_t<5> T_tv)

{
    DeviceMSELossForward5d<FLOAT>(I, T, lsum, divisor, I_tv, T_tv);
}

extern "C" __global__ void MSELossBackward5d(const FLOAT* __restrict__ I,
                                             const FLOAT* __restrict__ T,
                                             const FLOAT* __restrict__ dO,
                                             FLOAT* __restrict__ dI,
                                             FLOAT* __restrict__ dT,
                                             float divisor,
                                             tensor_view_t<5> I_tv,
                                             tensor_view_t<5> T_tv,
                                             tensor_view_t<5> dO_tv,
                                             tensor_view_t<5> dI_tv,
                                             tensor_view_t<5> dT_tv)
{
    DeviceMSELossBackward5d<FLOAT>(I, T, dO, dI, dT, divisor, I_tv, T_tv, dO_tv, dI_tv, dT_tv);
}
