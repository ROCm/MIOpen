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

inline int GetOffset(int max_diag_len, int d, int M, int N, miopenMatrixDiagAlignMode_t align)
{
    if(((align == MIOPEN_MATRIX_ALIGN_RIGHT_LEFT || align == MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT) &&
        d >= 0) ||
       ((align == MIOPEN_MATRIX_ALIGN_LEFT_RIGHT || align == MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT) &&
        d <= 0))
    {
        return max_diag_len - std::min(N - std::max(d, 0), M + std::min(d, 0));
    }
    return 0;
}

template <typename TIO>
void cpu_matrix_set_diag(const tensor<TIO> input,
                         const tensor<TIO> diag,
                         tensor<TIO>& ref_output,
                         const int64_t k0,
                         const int64_t k1,
                         const bool is_fwd,
                         const miopenMatrixDiagAlignMode_t align)
{
    auto size = ref_output.desc.GetElementSize();

    auto M = ref_output.desc.GetLengths()[ref_output.desc.GetNumDims() - 2];
    auto N = ref_output.desc.GetLengths()[ref_output.desc.GetNumDims() - 1];

    ford(size)([&](size_t gid) {
        int batch_id     = gid / M / N;
        int m            = (gid / N) % M;
        int n            = gid % N;
        int max_diag_len = std::min(M + std::min(k1, 0L), N + std::min(-k0, 0L));

        TIO input_val = input.desc.GetElementSize() > 0 ? input[0] : static_cast<TIO>(0);
        TIO val;
        if(k0 == k1)
        {
            if(n - m == k1)
            {
                int diag_id = batch_id * max_diag_len + n - std::max(k1, 0L);
                val         = is_fwd ? diag[diag_id] : 0;
            }
            else
            {
                val = input.desc.GetElementSize() > 0
                          ? (input.desc.GetElementSize() == 1 ? input_val : input[gid])
                          : 0;
            }
        }
        else
        {
            int d = n - m;
            if(k0 <= d && d <= k1)
            {
                int num_diags     = k1 - k0 + 1;
                int diag_index    = k1 - d;
                int offset        = GetOffset(max_diag_len, d, M, N, align);
                int index_in_diag = n - std::max(d, 0) + offset;
                int diag_id =
                    batch_id * num_diags * max_diag_len + diag_index * max_diag_len + index_in_diag;
                val = is_fwd ? diag[diag_id] : 0;
            }
            else
            {
                val = input.desc.GetElementSize() > 0
                          ? (input.desc.GetElementSize() == 1 ? input_val : input[gid])
                          : 0;
            }
        }
        ref_output[gid] = val;
    });
}
