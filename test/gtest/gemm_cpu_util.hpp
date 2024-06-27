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

namespace miopen
{
    namespace gemm_cpu_util
    {
        template <typename T>
        void CallGemm(miopen::GemmDescriptor gemm_desc,
                      Data_t A,
                      std::size_t a_offset,
                      Data_t B,
                      std::size_t b_offset,
                      Data_t C,
                      std::size_t c_offset)
        {
            const T* a_ptr = static_cast<const T*>(A);
            const T* b_ptr = static_cast<const T*>(B);
            T* c_ptr       = static_cast<T*>(C);

             // our cpu GEMM logic is row-major
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

            for(int bi = 0; bi < gemm_desc.batch_count; ++bi)
            {
                for(int mi = 0; mi < gemm_desc.m; ++mi)
                {
                    for(int ni = 0; ni < gemm_desc.n; ++ni)
                    {
                        double y = 0;
                        for(int ki = 0; ki < gemm_desc.k; ++ki)
                        {
                            int aindex = gemm_desc.transA ? a_offset + gemm_desc.strideA * bi + gemm_desc.lda * ki + mi
                                                          : a_offset + gemm_desc.strideA * bi + gemm_desc.lda * mi + ki;
                            int bindex = gemm_desc.transB ? b_offset + gemm_desc.strideB * bi + gemm_desc.ldb * ni + ki
                                                          : b_offset + gemm_desc.strideB * bi + gemm_desc.ldb * ki + ni;
                            y += static_cast<double>(a_ptr[aindex]) * static_cast<double>(b_ptr[bindex]);
                        }
                        int cindex = c_offset + gemm_desc.strideC * bi + gemm_desc.ldc * mi + ni;
                        c_ptr[cindex] =
                            static_cast<T>(static_cast<double>(gemm_desc.alpha) * y +
                                        static_cast<double>(gemm_desc.beta) * static_cast<double>(c_ptr[cindex]));
                    }
                }
            }
        }
    }
}
