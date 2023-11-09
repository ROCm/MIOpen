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
#include <cassert>
#include <miopen/tensor.hpp>
#include <miopen/matrixOps.hpp>
#include <miopen/logger.hpp>

namespace miopen {
namespace matrixOps {

GemmDescriptor::GemmDescriptor()
    : m_val(0),
      n_val(0),
      k_val(0),
      strideA(0),
      strideB(0),
      strideC(0),
      isColMajor(false),
      transA(false),
      transB(false)
{
}

GemmDescriptor::GemmDescriptor(long long int m_,
                               long long int n_,
                               long long int k_,
                               long long int  strideA_,
                               long long int  strideB_,
                               long long int  strideC_,
                               bool isColMajor_ = true,
                               bool transA_     = false,
                               bool transB_     = false)
    : m_val(m_),
      n_val(n_),
      k_val(k_),
      strideA(strideA_),
      strideB(strideB_),
      strideC(strideC_),
      isColMajor(isColMajor_),
      transA(transA_),
      transB(transB_)
{
}

bool GemmDescriptor::GetIsColMajor() const { return this->isColMajor; }
bool GemmDescriptor::IsTransA() const { return this->transA; }
bool GemmDescriptor::IsTransB() const { return this->transB; }

long long int GemmDescriptor::GetStrideA() const { return this->strideA; }
long long int GemmDescriptor::GetStrideB() const { return this->strideB; }
long long int GemmDescriptor::GetStrideC() const { return this->strideC; }

long long int GemmDescriptor::GetM() const { return this->m_val; }
long long int GemmDescriptor::GetN() const { return this->n_val; }
long long int GemmDescriptor::GetK() const { return this->k_val; }

void GemmDescriptor::SetIsColMajor(bool icm) { this->isColMajor = icm; }

std::ostream& operator<<(std::ostream& stream, const GemmDescriptor& gemm_desc)
{
    return stream << "{"
                  << "isColMajor " << gemm_desc.isColMajor << ", "
                  << "transA " << gemm_desc.transA << ", "
                  << "transB " << gemm_desc.transB << ", "
                  << "strideA " << gemm_desc.strideA << ", "
                  << "strideB " << gemm_desc.strideB << ", "
                  << "strideC " << gemm_desc.strideC << ", "
                  << "M " << gemm_desc.m_val << ", "
                  << "N " << gemm_desc.n_val << ", "
                  << "K " << gemm_desc.k_val << "} ";
}

TensorDescriptor GemmDescriptor::GetOutputTensor(const TensorDescriptor& ADesc,
                                                 const TensorDescriptor& BDesc) const
{
    if(ADesc.GetType() != BDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm, "Types do not match for matrix");
    }
    const auto& a_lens = ADesc.GetLengths(); // [M, K]
    const auto& b_lens = BDesc.GetLengths(); // [K, N]
    assert(a_lens.size() == 2 && b_lens.size() == 2);

    if(a_lens[1] != b_lens[0])
    {
        MIOPEN_THROW(miopenStatusBadParm, "Matrix cannot be multiplied because of dimension issue");
    }
    return {ADesc.GetType(), miopenTensorRowMajor, {a_lens[0], b_lens[1]}, {1, a_lens[0]}};
}

MatrixAdditionDescriptor::MatrixAdditionDescriptor(long long int m_,
                                                   long long int n_,
                                                   long long int k_,
                                                   long long int  strideC_,
                                                   long long int  strideD_,
                                                   long long int  strideE_,
                                                   bool isColMajor_ = true,
                                                   bool transC_     = false,
                                                   bool transD_     = false)
    : m_val(m_),
      n_val(n_),
      k_val(k_),
      strideC(strideC_),
      strideD(strideD_),
      strideE(strideE_),
      isColMajor(isColMajor_),
      transC(transC_),
      transD(transD_)
{
}

bool MatrixAdditionDescriptor::GetIsColMajor() const { return this->isColMajor; }
bool MatrixAdditionDescriptor::IsTransC() const { return this->transC; }
bool MatrixAdditionDescriptor::IsTransD() const { return this->transD; }

long long int MatrixAdditionDescriptor::GetStrideC() const { return this->strideC; }
long long int MatrixAdditionDescriptor::GetStrideD() const { return this->strideD; }
long long int MatrixAdditionDescriptor::GetStrideE() const { return this->strideE; }

long long int MatrixAdditionDescriptor::GetM() const { return this->m_val; }
long long int MatrixAdditionDescriptor::GetN() const { return this->n_val; }
long long int MatrixAdditionDescriptor::GetK() const { return this->k_val; }

void MatrixAdditionDescriptor::SetIsColMajor(bool icm) { this->isColMajor = icm; }

std::ostream& operator<<(std::ostream& stream,
                         const MatrixAdditionDescriptor& mat_add_desc)
{
    return stream << "{"
                  << "isColMajor " << mat_add_desc.isColMajor << ", "
                  << "transC " << mat_add_desc.transC << ", "
                  << "transD " << mat_add_desc.transD << ", "
                  << "strideC " << mat_add_desc.strideC << ", "
                  << "strideD " << mat_add_desc.strideD << ", "
                  << "strideE " << mat_add_desc.strideE << ", "
                  << "M " << mat_add_desc.m_val << ", "
                  << "N " << mat_add_desc.n_val << ", "
                  << "K " << mat_add_desc.k_val << "} ";
}

} // namespace matrixOps
} // namespace miopen
