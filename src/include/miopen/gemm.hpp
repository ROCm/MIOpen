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
#ifndef GUARD_MIOPEN_GEMM_HPP_
#define GUARD_MIOPEN_GEMM_HPP_

#include <miopen/gemm_geometry.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

#if MIOPEN_USE_MIOPENGEMM
// get rid of this
GemmGeometry CreateMIOpenGemmGeometry(bool isColMajor,
                                      bool transA,
                                      bool transB,
                                      int m,
                                      int n,
                                      int k,
                                      int lda,
                                      int ldb,
                                      int ldc,
                                      float alpha,
                                      float beta);
#endif

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
//     cross-row    memory stride for row   -major A, B, C.
// strideA, strideB, strideC are the strides of the matrices
void CallGemm(Handle& handle,
              bool isColMajor,
              bool transA,
              bool transB,
              int m,
              int n,
              int k,
              const void* alpha,
              ConstData_t A,
              int a_offset,
              int lda,
              ConstData_t B,
              int b_offset,
              int ldb,
              const void* beta,
              Data_t C,
              int c_offset,
              int ldc);

void CallGemmStridedBatched(Handle& handle,
                            bool isColMajor,
                            bool transA,
                            bool transB,
                            int m,
                            int n,
                            int k,
                            const void* alpha,
                            ConstData_t A,
                            int a_offset,
                            int lda,
                            long long int strideA,
                            ConstData_t B,
                            int b_offset,
                            int ldb,
                            long long int strideB,
                            const void* beta,
                            Data_t C,
                            int c_offset,
                            int ldc,
                            long long int strideC,
                            int batch_count);

// GEMM description for Convolution Fwd (non-batched)
// y = w * Im2Col(x)
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvFwd(const TensorDescriptor& wDesc,
                             const TensorDescriptor& xDesc,
                             const TensorDescriptor& yDesc);

// GEMM description for Convolution Bwd (non-batched)
// dx = Col2Im(transpose(w) * dy)
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvBwdData(const TensorDescriptor& wDesc,
                                 const TensorDescriptor& dyDesc,
                                 const TensorDescriptor& dxDesc);

// GEMM description for Convolution Bwd-Weight (non-batched)
// dw = dy * transpose(Im2Col(x))
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvBwdWeight(const TensorDescriptor& dyDesc,
                                   const TensorDescriptor& xDesc,
                                   const TensorDescriptor& dwDesc);

// GEMM description for 1x1 Convolution Fwd (non-batched)
// y = w * x
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConv1x1Fwd(const TensorDescriptor& wDesc,
                                const TensorDescriptor& xDesc,
                                const TensorDescriptor& yDesc);

// GEMM description for 1x1 Convolution Bwd-Data (non-batched)
// dx = transpose(w) * dy
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConv1x1BwdData(const TensorDescriptor& wDesc,
                                    const TensorDescriptor& dyDesc,
                                    const TensorDescriptor& dxDesc);

// GEMM description for 1x1 Convolution Bwd-Weight (non-batched)
// dw = dy * transpose(x)
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConv1x1BwdWeight(const TensorDescriptor& dyDesc,
                                      const TensorDescriptor& xDesc,
                                      const TensorDescriptor& dwDesc);

// GEMM description for 1x1 Convolution (using CNHW) Fwd (non-batched)
// y = CNHW2NCHW(w * NCHW2CNHW(x))
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvCNHWFwd(const TensorDescriptor& wDesc,
                                 const TensorDescriptor& xDesc,
                                 const TensorDescriptor& yDesc);

// GEMM description for 1x1 Convolution (using CNHW) Bwd-Data (non-batched)
// dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
std::tuple<bool, bool, bool, int, int, int, int, int, int, float, float>
CreateGemmDescriptionConvCNHWBwdData(const TensorDescriptor& wDesc,
                                     const TensorDescriptor& dyDesc,
                                     const TensorDescriptor& dxDesc);

// GEMM description for 1x1 Convolution Fwd (batched)
// y = w * x
std::tuple<bool,
           bool,
           bool,
           int,
           int,
           int,
           int,
           int,
           int,
           long long int,
           long long int,
           long long int,
           int,
           float,
           float>
CreateGemmStridedBatchedDescriptionConv1x1Fwd(const TensorDescriptor& wDesc,
                                              const TensorDescriptor& xDesc,
                                              const TensorDescriptor& yDesc);

// GEMM description for 1x1 Convolution Bwd-Data (batched)
// dx = transpose(w) * dy
std::tuple<bool,
           bool,
           bool,
           int,
           int,
           int,
           int,
           int,
           int,
           long long int,
           long long int,
           long long int,
           int,
           float,
           float>
CreateGemmStridedBatchedDescriptionConv1x1BwdData(const TensorDescriptor& wDesc,
                                                  const TensorDescriptor& dyDesc,
                                                  const TensorDescriptor& dxDesc);

} // namespace miopen

namespace miopen {

GemmGeometry
GetGemmGeometry(Handle& handle, std::string algorithm_name, std::string network_config);

GemmGeometry CreateGemmGeometryTranBwdData(const TensorDescriptor& dyDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& dxDesc,
                                           bool isDataColMajor,
                                           std::string& network_config);

GemmGeometry CreateGemmGeometryConvBwdWeights(const TensorDescriptor& dyDesc,
                                              const TensorDescriptor& xDesc,
                                              const TensorDescriptor& dwDesc,
                                              bool isDataColMajor,
                                              std::string& network_config);

GemmGeometry CreateGemmGeometryConvBwdData(const TensorDescriptor& dyDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& dxDesc,
                                           bool isDataColMajor,
                                           std::string& network_config);

GemmGeometry CreateGemmGeometryConvBwdDataCNHW(const TensorDescriptor& dyDesc,
                                               const TensorDescriptor& wDesc,
                                               const TensorDescriptor& dxDesc,
                                               bool isDataColMajor,
                                               std::string& network_config);

GemmGeometry CreateGemmGeometryConvFwd(const TensorDescriptor& xDesc,
                                       const TensorDescriptor& wDesc,
                                       const TensorDescriptor& yDesc,
                                       bool isDataColMajor,
                                       std::string& network_config);

GemmGeometry CreateGemmGeometryConvFwdCNHW(const TensorDescriptor& xDesc,
                                           const TensorDescriptor& wDesc,
                                           const TensorDescriptor& yDesc,
                                           bool isDataColMajor,
                                           std::string& network_config);

GemmGeometry CreateGemmGeometryRNN(int M,
                                   int N,
                                   int K,
                                   float alpha,
                                   float beta,
                                   bool tA,
                                   bool tB,
                                   bool tC,
                                   int lda,
                                   int ldb,
                                   int ldc,
                                   bool isDataColMajor,
                                   std::string& network_config);

GemmGeometry ScanGemmGeometryRNN(Handle& handle,
                                 ConstData_t A,
                                 ConstData_t B,
                                 Data_t C,
                                 int M,
                                 int N,
                                 int K,
                                 float alpha,
                                 float beta,
                                 bool tA,
                                 bool tB,
                                 bool tC,
                                 int lda,
                                 int ldb,
                                 int ldc,
                                 bool isDataColMajor,
                                 std::string& network_config,
                                 float timeout);

void RunGemmGeometryRNN(Handle& handle,
                        ConstData_t A,
                        ConstData_t B,
                        Data_t C,
                        int M,
                        int N,
                        int K,
                        float alpha,
                        float beta,
                        bool tA,
                        bool tB,
                        bool tC,
                        int lda,
                        int ldb,
                        int ldc,
                        int a_offset,
                        int b_offset,
                        int c_offset,
                        bool isDataColMajor,
                        std::string& network_config,
                        float timeout);
} // namespace miopen

#endif // GUARD_MIOPEN_GEMM_HPP_
