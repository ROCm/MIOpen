/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_GEMM_HPP_
#define MIOPEN_GEMM_HPP_

#include <miopen/tensor.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>

namespace miopen {
namespace matrixOps {

struct GemmDescriptor : miopenGemmDescriptor
{
    GemmDescriptor();
    GemmDescriptor(long long int m_,
                   long long int n_,
                   long long int k_,
                   long long int strideA_,
                   long long int strideB_,
                   long long int strideC_,
                   bool isColMajor_,
                   bool transA_,
                   bool transB_);

    bool GetIsColMajor() const;
    bool IsTransA() const;
    bool IsTransB() const;
    long long int GetM() const;
    long long int GetN() const;
    long long int GetK() const;
    long long int GetStrideA() const;
    long long int GetStrideB() const;
    long long int GetStrideC() const;

    void SetIsColMajor(bool);

    // stream out operator overloading for MIOpen log functions
    friend std::ostream& operator<<(std::ostream& stream, const GemmDescriptor& x);

    TensorDescriptor GetOutputTensor(const TensorDescriptor& ADesc,
                                     const TensorDescriptor& BDesc) const;

private:
    long long int m_val, n_val, k_val;
    long long int strideA, strideB, strideC;

    bool isColMajor;     // these are not in tensor
    bool transA, transB; // these are not in tensor
};

// E = C + D
struct MatrixAdditionDescriptor : miopenMatrixAdditionDescriptor
{
    MatrixAdditionDescriptor(long long int m_,
                             long long int n_,
                             long long int k_,
                             long long int strideC_,
                             long long int strideD_,
                             long long int strideE_,
                             bool isColMajor_,
                             bool transC_,
                             bool transD_);

    bool GetIsColMajor() const;
    bool IsTransC() const;
    bool IsTransD() const;
    long long int GetM() const;
    long long int GetN() const;
    long long int GetK() const;
    long long int GetStrideC() const;
    long long int GetStrideD() const;
    long long int GetStrideE() const;

    void SetIsColMajor(bool);

    // stream out operator overloading for MIOpen log functions
    friend std::ostream& operator<<(std::ostream& stream, const MatrixAdditionDescriptor& x);

private:
    long long int m_val, n_val, k_val;
    long long int strideC, strideD, strideE;

    bool isColMajor;     // these are not in tensor
    bool transC, transD; // these are not in tensor
};

} // namespace matrixOps
} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenGemmDescriptor, miopen::matrixOps::GemmDescriptor);
MIOPEN_DEFINE_OBJECT(miopenMatrixAdditionDescriptor, miopen::matrixOps::MatrixAdditionDescriptor);
#endif // MIOPEN_GEMM_HPP_
