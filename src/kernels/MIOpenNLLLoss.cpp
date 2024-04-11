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
#include <cstddef>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

#if MIOPEN_USE_BFP16 == 1
#define CVT_FLOAT2ACCUM(x) (bfloat16_to_float(x))
#define CVT_ACCUM2FLOAT(x) (float_to_bfloat16(x))
#define CVT_INTEGRAL2ACCUM(x) ((_FLOAT_ACCUM)(x))
#define CVT_FP32_2FLOAT(x) (CVT_ACCUM2FLOAT(x))
#define CVT_FP32_2ACCUM(x) (x)
#endif

/* input(input): [N, C, D1, D2], target(target): [N, D1, D2],
 * weight(weight): [C], output(output): [N, D1, D2] */
/* Each thread computes one output: output[n0][n1][n2] */
extern "C" __global__ void NLLLossUnreducedForward4dContiguous(const FLOAT_ACCUM* __restrict__ input, 
                                                               const FLOAT_ACCUM* __restrict__ target, 
                                                               const FLOAT_ACCUM* weight,
                                                               FLOAT_ACCUM* __restrict__ output, 
                                                               long ignore_index, 
                                                               size_t N_total,
                                                               size_t C,
                                                               size_t D1,
                                                               size_t D2)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (gid >= N_total) return;

    size_t NWH[3];
    NWH[2] = (gid) % D2;
    size_t nc = (gid) / D2;
    NWH[1] = nc % D1;
    NWH[0] = nc / D1;

    long t = static_cast<long>(target[gid]);
    // t: Class index
    if (t < 0 || t == ignore_index || t >= C)  {
        output[gid] = 0;
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? weight[t] : 1.0f;

    // fix this
    // FLOAT_ACCUM input_value = input[NWH[0]][t][NWH[1]][NWH[2]];
    FLOAT_ACCUM input_value = input[(NWH[0] * C + t) * D1 * D2 + NWH[1] * D2 + NWH[2]];

    output[gid] = -1.0f * w * input_value;
}
