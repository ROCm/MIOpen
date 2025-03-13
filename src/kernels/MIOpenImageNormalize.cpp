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

template <typename TIO>
__device__ void DeviceImageNormalizeFwdContiguous(const TIO* input,
                                                  const TIO* mean,
                                                  const TIO* std,
                                                  TIO* output,
                                                  size_t c_stride,
                                                  size_t C,
                                                  size_t N)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= N)
        return;

    size_t c = gid / c_stride % C;

    FLOAT_ACCUM pixel  = CVT_FLOAT2ACCUM(input[gid]);
    FLOAT_ACCUM mean_p = CVT_FLOAT2ACCUM(mean[c]);
    FLOAT_ACCUM std_p  = CVT_FLOAT2ACCUM(std[c]);
    FLOAT_ACCUM result = (pixel - mean_p) / std_p;

    output[gid] = CVT_ACCUM2FLOAT(result);
}

extern "C" __global__ void ImageNormalizeContiguous(const DTYPE* input,
                                                    const DTYPE* mean,
                                                    const DTYPE* std,
                                                    DTYPE* output,
                                                    size_t c_stride,
                                                    size_t C,
                                                    size_t N)
{
    DeviceImageNormalizeFwdContiguous<DTYPE>(input, mean, std, output, c_stride, C, N);
}