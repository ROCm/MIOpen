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

// Workaround to overcome redefinition errors while including rocrand header files directly
#include "miopen_rocrand.hpp"

// NOTE: This function is identical to `InitKernelStateHIP` in src/kernels/MIOpenDropoutHIP.cpp.
// In the future, these should be refactored to retain only one implementation.
// For now, both are kept to minimize changes, as this PR only introduces `GenerateRandomBitMask`
// without refactoring Dropout.
extern "C" __global__ void
InitKernelStateHIP(rocrand_state_xorwow* state, ulong prng_seed, ulong states_num)
{
    // Get the index of the current element
    size_t index  = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t gid = index; gid < states_num; gid += stride)
    {
        rocrand_state_xorwow state_gid;
        rocrand_init(prng_seed, gid, 0ULL, &state_gid);
        state[gid] = state_gid;
    }
}
