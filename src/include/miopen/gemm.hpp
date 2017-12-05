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

GemmGeometry GetGemmGeometry(std::string algorithm_name, std::string network_config);

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

GemmGeometry CreateGemmGeometryConvFwd(const TensorDescriptor& xDesc,
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

GemmGeometry CreateMIOpenGemmGeometry(int M,
                                      int N,
                                      int K,
                                      int lda,
                                      int ldb,
                                      int ldc,
                                      bool tA,
                                      bool tB,
                                      bool isDataColMajor,
                                      float alpha,
                                      float beta);

} // namespace miopen

#endif // GUARD_MIOPEN_GEMM_HPP_
