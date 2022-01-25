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
#include <cassert>
#include <miopen/gemm.hpp>
#include <miopen/logger.hpp>

namespace miopen {

GemmNewDescriptor::GemmNewDescriptor() {}

GemmNewDescriptor::GemmNewDescriptor(
                   bool isColMajor_,
                   bool transA_,
                   bool transB_,
                   int m_,
                   int n_, 
                   int k_,
                   int lda_,
                   int ldb_,
                   int ldc_,
                   long long int strideA_,
                   long long int strideB_,
                   long long int strideC_,
                   double alpha_,
                   double beta_,
                   int batch_count_,
                   miopenDataType_t dataType_)
    :isColMajor(isColMajor_),
      transA(transA_),
      transB(transB_),
      m(m_),
      n(n_),
      k(k_),
      lda(lda_),
      ldb(ldb_),
      ldc(ldc_),
      strideA(strideA_),
      strideB(strideB_),
      strideC(strideC_),
      alpha(alpha_),
      beta(beta_),
      batch_count(batch_count_),
      dataType(dataType_)
{
}

bool GemmNewDescriptor::GetIsColMajor() const { return this->isColMajor; }

bool GemmNewDescriptor::GetTransA() const { return this->transA; }

bool GemmNewDescriptor::GetTransB() const { return this->transB; }

int GemmNewDescriptor::GetM() const { return this->m; }

int GemmNewDescriptor::GetN() const { return this->n; }

int GemmNewDescriptor::GetK() const { return this->k; }

int GemmNewDescriptor::Getlda() const { return this->lda; }

int GemmNewDescriptor::Getldb() const { return this->ldb; }

int GemmNewDescriptor::Getldc() const { return this->ldc; }

long long int GemmNewDescriptor::GetStrideA() const { return this->strideA; }

long long int GemmNewDescriptor::GetStrideB() const { return this->strideB; }

long long int GemmNewDescriptor::GetStrideC() const { return this->strideC; }

double GemmNewDescriptor::GetAlpha() const { return this->alpha; }

double GemmNewDescriptor::GetBeta() const { return this->beta; }

int GemmNewDescriptor::GetBatchCount() const { return this->batch_count; }

miopenDataType_t GemmNewDescriptor::GetMIOpenDataType() const { return this->dataType; }

std::ostream& operator<<(std::ostream& stream, const GemmNewDescriptor& gemm_desc)
{
    return stream << "{"
                  << "isColMajor " << gemm_desc.isColMajor << ", "
                  << "transA " << gemm_desc.transA << ", "
                  << "transB " << gemm_desc.transB << ", "
                  << "m " << gemm_desc.m << ", "
                  << "n " << gemm_desc.n << ", "
                  << "k " << gemm_desc.k << ", "
                  << "lda " << gemm_desc.lda << ", "
                  << "ldb " << gemm_desc.ldb << ", "
                  << "ldc " << gemm_desc.ldc << ", "
                  << "batch_count " << gemm_desc.batch_count << ", "
                  << "strideA " << gemm_desc.strideA << ", "
                  << "strideB " << gemm_desc.strideB << ", "
                  << "strideC " << gemm_desc.strideC << ", "
                  << "alpha " << gemm_desc.alpha << ", "
                  << "beta " << gemm_desc.beta << ", "
                  << "dataType " << gemm_desc.dataType << "} ";
}

} // namespace miopen
