/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <hip/hip_runtime.h>
#endif

extern "C" __global__ __launch_bounds__(256, 2) void wrw_reduction_hip(
    float* output, float* input, int out_length, int in_stride, int n_groups)
{
    float4 vec_in;
    float4 vec_out;
    int i_len, i_groups;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    int offset       = bid * out_length * 256 + tid * out_length;

    float* local_in  = input + offset;
    float* local_out = output + offset;

    for(i_len = 0; i_len < out_length; i_len += 4)
    {
        vec_out = (float4)0;
        for(i_groups = 0; i_groups < n_groups; i_groups++)
        {
            vec_in = *(float4*)(local_in + i_len + in_stride * i_groups);
            vec_out += vec_in;
        }
        *(float4*)(local_out + i_len) = vec_out;
    }
}
