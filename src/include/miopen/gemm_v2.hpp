/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_GEMM_V2_HPP_
#define GUARD_MIOPEN_GEMM_V2_HPP_

#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

// GEMM operation: C = alpha * op(A) * op(B) + beta * C.
// op() can be either transpose or no-operation for A or B.
// The shape (nRow x nCol) of op(A), op(B), C are:
//   m x n,
//   n x k,
//   m x n.
// A, B, C are what are actually being saved in memory,
//   they can either be all column-major or all row-major.
// lda, ldb, ldc are leading dimension strides of memory for A, B, C,
//   and leading dimension stride is:
//     cross-column memory stride for column-major A, B, C,
//     cross-row    memory stride for row   -major A, B, C
// for strided batched GEMM
//   strideA, strideB, strideC are the strides of the matrices
struct GemmDescriptor
{
    bool isColMajor;
    bool transA, transB;
    int m, n, k;
    int lda, ldb, ldc;
    int batch_count;
    long long int strideA, strideB, strideC;
    float alpha, beta;
};

void CallGemm(Handle& handle,
              GemmDescriptor gemm_desc,
              ConstData_t A,
              int a_offset,
              ConstData_t B,
              int b_offset,
              Data_t C,
              int c_offset,
              std::string* kcache_key = nullptr);

void CallGemmStridedBatched(Handle& handle,
                            GemmDescriptor gemm_desc,
                            ConstData_t A,
                            int a_offset,
                            ConstData_t B,
                            int b_offset,
                            Data_t C,
                            int c_offset,
                            std::string* kcache_key = nullptr);

void CallGemmStridedBatchedSequential(Handle& handle,
                                      GemmDescriptor gemm_desc,
                                      ConstData_t A,
                                      int a_offset,
                                      ConstData_t B,
                                      int b_offset,
                                      Data_t C,
                                      int c_offset,
                                      std::string* kcache_key = nullptr);

// GEMM parameters for Convolution (using Im2Col) Fwd
// y = w * Im2Col(x)
GemmDescriptor CreateGemmDescriptorConvFwd(const TensorDescriptor& wDesc,
                                           const TensorDescriptor& xDesc,
                                           const TensorDescriptor& yDesc);

// GEMM parameters for Convolution (using Im2Col) Bwd-Data
// dx = Col2Im(transpose(w) * dy)
GemmDescriptor CreateGemmDescriptorConvBwdData(const TensorDescriptor& wDesc,
                                               const TensorDescriptor& dyDesc,
                                               const TensorDescriptor& dxDesc);

// GEMM parameters for Convolution (using Im2Col) Bwd-Weight
// dw = dy * transpose(Im2Col(x))
GemmDescriptor CreateGemmDescriptorConvBwdWeight(const TensorDescriptor& dyDesc,
                                                 const TensorDescriptor& xDesc,
                                                 const TensorDescriptor& dwDesc);

// GEMM parameters for 1x1 Convolution (using CNHW) Fwd
// y = CNHW2NCHW(w * NCHW2CNHW(x))
GemmDescriptor CreateGemmDescriptorConvCNHWFwd(const TensorDescriptor& wDesc,
                                               const TensorDescriptor& xDesc,
                                               const TensorDescriptor& yDesc);

// GEMM parameters for 1x1 Convolution (using CNHW) Bwd-Data
// dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
GemmDescriptor CreateGemmDescriptorConvCNHWBwdData(const TensorDescriptor& wDesc,
                                                   const TensorDescriptor& dyDesc,
                                                   const TensorDescriptor& dxDesc);

// strided batched GEMM parameters for 1x1 Convolution Fwd
// y[i] = w * x[i], i is batch id
GemmDescriptor CreateGemmStridedBatchedParamConv1x1Fwd(const TensorDescriptor& wDesc,
                                                       const TensorDescriptor& xDesc,
                                                       const TensorDescriptor& yDesc);

// strided batched GEMM parameters for 1x1 Convolution Bwd-Data
// dx[i] = transpose(w) * dy[i], i is batch id
GemmDescriptor CreateGemmStridedBatchedParamConv1x1BwdData(const TensorDescriptor& wDesc,
                                                           const TensorDescriptor& dyDesc,
                                                           const TensorDescriptor& dxDesc);

// strided batched GEMM parameters for 1x1 Convolution Bwd-Weight
// dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
GemmDescriptor CreateGemmStridedBatchedParamConv1x1BwdWeight(const TensorDescriptor& dyDesc,
                                                             const TensorDescriptor& xDesc,
                                                             const TensorDescriptor& dwDesc);
} // namespace miopen

#endif // GUARD_MIOPEN_GEMM_V2_HPP_
