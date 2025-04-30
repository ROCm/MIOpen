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
#pragma once

#include <miopen/gemm_v2.hpp>
#include "../rnn_util.hpp"
namespace miopen {
namespace gemm_cpu_util {
template <typename T>
void CallGemm(miopen::GemmDescriptor gemm_desc,
              Data_t A,
              std::size_t a_offset,
              Data_t B,
              std::size_t b_offset,
              Data_t C,
              std::size_t c_offset)
{
    const T* a_ptr = static_cast<const T*>(A) + a_offset;
    const T* b_ptr = static_cast<const T*>(B) + b_offset;
    T* c_ptr       = static_cast<T*>(C) + c_offset;

    if(gemm_desc.isColMajor)
    {
        gemm_desc.isColMajor = false;
        std::swap(a_ptr, b_ptr);
        std::swap(a_offset, b_offset);
        std::swap(gemm_desc.transA, gemm_desc.transB);
        std::swap(gemm_desc.m, gemm_desc.n);
        std::swap(gemm_desc.lda, gemm_desc.ldb);
        std::swap(gemm_desc.strideA, gemm_desc.strideB);
    }

    size_t a_col = gemm_desc.transA ? gemm_desc.m : gemm_desc.k;
    size_t a_row = gemm_desc.transA ? gemm_desc.k : gemm_desc.m;

    size_t b_col = gemm_desc.transB ? gemm_desc.k : gemm_desc.n;
    size_t b_row = gemm_desc.transB ? gemm_desc.n : gemm_desc.k;

    RNN_mm_cpu_batched<T>(a_ptr,
                          a_col,
                          a_row,
                          gemm_desc.lda,
                          gemm_desc.strideA,
                          gemm_desc.transA,
                          b_ptr,
                          b_col,
                          b_row,
                          gemm_desc.ldb,
                          gemm_desc.strideB,
                          gemm_desc.transB,
                          c_ptr,
                          gemm_desc.n,
                          gemm_desc.m,
                          gemm_desc.ldc,
                          gemm_desc.strideC,
                          0,
                          gemm_desc.batch_count,
                          gemm_desc.alpha,
                          gemm_desc.beta);
}
} // namespace gemm_cpu_util
} // namespace miopen
