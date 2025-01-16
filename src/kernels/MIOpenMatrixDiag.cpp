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

enum class MatrixAlignMode_t
{
    LEFT_RIGHT = 0,
    RIGHT_LEFT,
    LEFT_LEFT,
    RIGHT_RIGHT,
};

#ifndef __HIP_DEVICE_COMPILE__
static_assert(MIOPEN_MATRIX_ALIGN_LEFT_RIGHT == static_cast<int>(MatrixAlignMode_t::LEFT_RIGHT));
static_assert(MIOPEN_MATRIX_ALIGN_RIGHT_LEFT == static_cast<int>(MatrixAlignMode_t::RIGHT_LEFT));
static_assert(MIOPEN_MATRIX_ALIGN_LEFT_LEFT == static_cast<int>(MatrixAlignMode_t::LEFT_LEFT));
static_assert(MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT == static_cast<int>(MatrixAlignMode_t::RIGHT_RIGHT));
#endif

inline __device__ int64_t
GetOffset(int64_t max_diag_len, int64_t d, int64_t M, int64_t N, MatrixAlignMode_t align)
{
    if(((align == MatrixAlignMode_t::RIGHT_LEFT || align == MatrixAlignMode_t::RIGHT_RIGHT) &&
        d >= 0) ||
       ((align == MatrixAlignMode_t::LEFT_RIGHT || align == MatrixAlignMode_t::RIGHT_RIGHT) &&
        d <= 0))
    {
        return max_diag_len - min(N - max(d, 0LL), M + min(d, 0LL));
    }
    return 0;
}

template <typename TIO, MatrixAlignMode_t ALIGN_T>
__device__ void MatrixSetDiag(const TIO* input,
                              const TIO* diagonal,
                              TIO* output,
                              const int64_t k0,
                              const int64_t k1,
                              const uint64_t M,
                              const uint64_t N,
                              const uint64_t numel,
                              const bool is_fwd,
                              const bool is_input_padding)
{
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lid = threadIdx.x;
    if(gid >= numel)
        return;

    int64_t batch_id     = gid / M / N;
    int64_t m            = (gid / N) % M;
    int64_t n            = gid % N;
    int64_t max_diag_len = min(M + min(k1, 0L), N + min(-k0, 0L));

    __shared__ TIO input_val;
    if(is_input_padding)
    {
        if(lid == 0)
            if(input)
                input_val = input[0];
            else
                input_val = 0;
        __syncthreads();
    }

    TIO val;
    if(k0 == k1)
    {
        if(n - m == k1)
        {
            int64_t diag_id = batch_id * max_diag_len + n - max(k1, 0LL);
            if(is_fwd)
                val = diagonal[diag_id];
            else
                val = 0;
        }
        else
        {
            if(input)
                val = (is_input_padding ? input_val : input[gid]);
            else
                val = 0;
        }
    }
    else
    {
        int64_t d = n - m;
        if(k0 <= d && d <= k1)
        {
            int64_t num_diags     = k1 - k0 + 1;
            int64_t diag_index    = k1 - d;
            int64_t offset        = GetOffset(max_diag_len, d, M, N, ALIGN_T);
            int64_t index_in_diag = n - max(d, 0LL) + offset;
            int64_t diag_id =
                batch_id * num_diags * max_diag_len + diag_index * max_diag_len + index_in_diag;
            if(is_fwd)
                val = diagonal[diag_id];
            else
                0;
        }
        else
        {
            if(input)
                val = (is_input_padding ? input_val : input[gid]);
            else
                val = 0;
        }
    }
    output[gid] = val;
}

extern "C" __global__ void MatrixSetDiag(const DTYPE* input,
                                         const DTYPE* diagonal,
                                         DTYPE* output,
                                         const int64_t k0,
                                         const int64_t k1,
                                         const uint64_t M,
                                         const uint64_t N,
                                         const uint64_t numel,
                                         const bool is_fwd,
                                         const bool is_input_padding)
{
    // instantiate the kernel
    MatrixSetDiag<DTYPE, static_cast<MatrixAlignMode_t>(ALIGN)>(
        input, diagonal, output, k0, k1, M, N, numel, is_fwd, is_input_padding);
}

template <typename TIO, MatrixAlignMode_t ALIGN_T>
__device__ void MatrixDiagPart(const TIO* input,
                               const TIO* padding,
                               TIO* output,
                               const int64_t k0,
                               const int64_t k1,
                               const uint64_t M,
                               const uint64_t N,
                               const uint64_t numel,
                               const bool is_single_padding)
{
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t lid = threadIdx.x;
    if(gid >= numel)
        return;

    int max_diag_len = min(M + min(k1, 0L), N + min(-k0, 0L));

    __shared__ TIO padding_val;
    if(is_single_padding)
    {
        if(lid == 0)
            if(padding)
                padding_val = padding[0];
            else
                padding_val = 0;
        __syncthreads();
    }

    TIO val;
    if(k0 == k1)
    {
        int64_t batch_id = gid / max_diag_len;
        int64_t n        = gid % max_diag_len;
        int64_t x        = max(k1, 0L);
        int64_t y        = max(-k1, 0L);
        if(0 <= n + y && n + y < M && 0 <= n + x && n + x < N)
        {
            uint64_t input_id = batch_id * M * N + (n + y) * N + n + x;
            val               = input[input_id];
        }
        else
        {
            val = padding_val;
        }
    }
    else
    {
        int64_t num_diags = k1 - k0 + 1;
        int64_t batch_id  = gid / num_diags / max_diag_len;
        int64_t m         = (gid / max_diag_len) % num_diags;
        int64_t n         = gid % max_diag_len;
        int64_t d         = k1 - m;
        int64_t offset    = GetOffset(max_diag_len, d, M, N, ALIGN_T);
        int64_t y         = max(-d, 0L) - offset;
        int64_t x         = max(d, 0L) - offset;

        if(0 <= n + y && n + y < M && 0 <= n + x && n + x < N)
        {
            uint64_t input_id = batch_id * M * N + (n + y) * N + n + x;
            val               = input[input_id];
        }
        else
        {
            val = padding_val;
        }
    }
    output[gid] = val;
}

extern "C" __global__ void MatrixDiagPart(const DTYPE* input,
                                          const DTYPE* padding,
                                          DTYPE* output,
                                          const int64_t k0,
                                          const int64_t k1,
                                          const uint64_t M,
                                          const uint64_t N,
                                          const uint64_t numel,
                                          const bool is_single_padding)
{
    // instantiate the kernel
    MatrixDiagPart<DTYPE, static_cast<MatrixAlignMode_t>(ALIGN)>(
        input, padding, output, k0, k1, M, N, numel, is_single_padding);
}
