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

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

// Workaround to overcome redefinition errors while including rocrand header files directly
#include "miopen_rocrand.hpp"

__device__ uchar vec_merge_bits(int* comp, int size)
{
    uchar result = 0;
    for(int i = 0; i < size; i++)
    {
        result |= (comp[i] & 1) << i;
    }
    return result;
}

template <uint32_t VLEN>
__device__ void
generate_random_bit_mask(rocrand_state_xorwow* states_in, uchar* mask, uint64_t N, float prob)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N)
        return;

    rocrand_state_xorwow cur_state; // Read the state of the current thread
    cur_state = states_in[gid];

    for(auto i = gid; i < N; i += blockDim.x * gridDim.x)
    {
        int rvals[VLEN];

#pragma unroll
        for(int j = 0; j < VLEN; j++)
        {
            auto random_fval = prng::xorwow_uniform(&cur_state);
            rvals[j]         = static_cast<int>(random_fval > prob);
        }

        mask[i] = vec_merge_bits(rvals, VLEN);
    }
}

extern "C" __global__ void
GenerateRandomBitMask(rocrand_state_xorwow* states_in, uchar* mask, uint64_t N, float prob)
{
    generate_random_bit_mask<VEC_LENGTH>(states_in, mask, N, prob);
}
