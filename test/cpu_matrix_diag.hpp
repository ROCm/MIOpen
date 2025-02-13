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
#pragma once

#include "tensor_holder.hpp"
#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

#include <cmath>

inline int64_t
GetOffset(int64_t max_diag_len, int64_t d, int64_t M, int64_t N, miopenMatrixDiagAlignMode_t align)
{
    if(((align == MIOPEN_MATRIX_ALIGN_RIGHT_LEFT || align == MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT) &&
        d >= 0) ||
       ((align == MIOPEN_MATRIX_ALIGN_LEFT_RIGHT || align == MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT) &&
        d <= 0))
    {
        return max_diag_len - std::min(N - std::max(d, static_cast<int64_t>(0)),
                                       M + std::min(d, static_cast<int64_t>(0)));
    }
    return 0;
}

template <typename TIO>
void cpu_matrix_set_diag_forward(const tensor<TIO> input,
                                 const tensor<TIO> diag,
                                 tensor<TIO>& ref_output,
                                 const int64_t k0,
                                 const int64_t k1,
                                 const bool is_fwd,
                                 const miopenMatrixDiagAlignMode_t align)
{
    size_t inSize  = input.desc.GetElementSize();
    size_t outSize = ref_output.desc.GetElementSize();

    size_t M = ref_output.desc.GetLengths()[ref_output.desc.GetNumDims() - 2];
    size_t N = ref_output.desc.GetLengths()[ref_output.desc.GetNumDims() - 1];

    size_t max_diag_len = std::min(M + std::min(k1, static_cast<int64_t>(0)),
                                   N + std::min(-k0, static_cast<int64_t>(0)));
    TIO input_val       = inSize > 0 ? input[0] : static_cast<TIO>(0);

    par_ford(outSize)([&](size_t gid) {
        int64_t batch_id = gid / M / N;
        int64_t m        = (gid / N) % M;
        int64_t n        = gid % N;

        TIO val;
        if(k0 == k1)
        {
            if(n - m == k1)
            {
                int64_t diag_id =
                    batch_id * max_diag_len + n - std::max(k1, static_cast<int64_t>(0));
                if(is_fwd)
                    val = diag[diag_id];
                else
                    val = 0;
            }
            else
            {
                if(inSize > 0)
                    val = inSize == 1 ? input_val : input[gid];
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
                int64_t offset        = GetOffset(max_diag_len, d, M, N, align);
                int64_t index_in_diag = n - std::max(d, static_cast<int64_t>(0)) + offset;
                int64_t diag_id =
                    batch_id * num_diags * max_diag_len + diag_index * max_diag_len + index_in_diag;
                if(is_fwd)
                    val = diag[diag_id];
                else
                    val = 0;
            }
            else
            {
                if(inSize > 0)
                    val = inSize == 1 ? input_val : input[gid];
                else
                    val = 0;
            }
        }
        ref_output[gid] = val;
    });
}

template <typename TIO>
void cpu_matrix_diag_part_forward(const tensor<TIO> input,
                                  const tensor<TIO> pad,
                                  tensor<TIO>& ref_output,
                                  const int64_t k0,
                                  const int64_t k1,
                                  const miopenMatrixDiagAlignMode_t align)
{
    auto padSize = pad.desc.GetElementSize();
    auto outSize = ref_output.desc.GetElementSize();

    auto M = input.desc.GetLengths()[input.desc.GetNumDims() - 2];
    auto N = input.desc.GetLengths()[input.desc.GetNumDims() - 1];

    int64_t max_diag_len = std::min(M + std::min(k1, static_cast<int64_t>(0)),
                                    N + std::min(-k0, static_cast<int64_t>(0)));
    TIO padding_val      = padSize > 0 ? pad[0] : static_cast<TIO>(0);

    par_ford(outSize)([&](size_t gid) {
        TIO val;
        if(k0 == k1)
        {
            int64_t batch_id = gid / max_diag_len;
            int64_t n        = gid % max_diag_len;
            int64_t x        = std::max(k1, static_cast<int64_t>(0));
            int64_t y        = std::max(-k1, static_cast<int64_t>(0));
            if(0 <= n + y && n + y < M && 0 <= n + x && n + x < N)
            {
                int64_t input_id = batch_id * M * N + (n + y) * N + n + x;
                val              = input[input_id];
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
            int64_t offset    = GetOffset(max_diag_len, d, M, N, align);
            int64_t y         = std::max(-d, static_cast<int64_t>(0)) - offset;
            int64_t x         = std::max(d, static_cast<int64_t>(0)) - offset;

            if(0 <= n + y && n + y < M && 0 <= n + x && n + x < N)
            {
                int64_t input_id = batch_id * M * N + (n + y) * N + n + x;
                val              = input[input_id];
            }
            else
            {
                val = padding_val;
            }
        }
        ref_output[gid] = val;
    });
}

template <typename TIO>
void cpu_matrix_set_diag_backward(const tensor<TIO> output_grad,
                                  tensor<TIO>& ref_input_grad,
                                  tensor<TIO>& ref_diag_grad,
                                  const int64_t k0,
                                  const int64_t k1,
                                  const miopenMatrixDiagAlignMode_t align)
{
    const auto& fake_diag = output_grad;
    cpu_matrix_set_diag_forward(output_grad, fake_diag, ref_input_grad, k0, k1, false, align);
    auto fake_pad = tensor<TIO>{{1}};
    fake_pad[0]   = 0;
    cpu_matrix_diag_part_forward(output_grad, fake_pad, ref_diag_grad, k0, k1, align);
}
