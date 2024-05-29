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

template <typename TI, typename TT, typename TO>
__device__ void softmarginlossunreducedforward5d(const TI* __restrict__ I,
                                                 const TT* __restrict__ T,
                                                 TO* __restrict__ O,
                                                 tensor_view_t<5> I_tv,
                                                 tensor_view_t<5> T_tv,
                                                 tensor_view_t<5> O_tv)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    tensor_layout_t<5> idx(I_tv, gid);
    if(idx.layout[0] >= I_tv.size[0])
        return;

    FLOAT_ACCUM i                    = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM t                    = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx(idx)]);
    O[O_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(log(1 + exp(-i * t)));
}

extern "C" __global__ void SoftMarginLossUnreducedForward5d(const INPUT_TYPE* __restrict__ I,
                                                            const TARGET_TYPE* __restrict__ T,
                                                            OUTPUT_TYPE* __restrict__ O,
                                                            tensor_view_t<5> I_tv,
                                                            tensor_view_t<5> T_tv,
                                                            tensor_view_t<5> O_tv)
{
    // instantiate the kernel
    softmarginlossunreducedforward5d<INPUT_TYPE, TARGET_TYPE, OUTPUT_TYPE>(
        I, T, O, I_tv, T_tv, O_tv);
}