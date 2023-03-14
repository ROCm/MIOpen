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

GemmDesc::GemmDesc() {}

GemmDesc::GemmDesc(int m_,
                   int n_,
                   int k_,
                   long long int lda_,
                   long long int ldb_,
                   long long int ldc_,
                   miopenDataType_t dataType_)
{
    this->isColMajor  = false;
    this->transA      = false;
    this->transB      = false;
    this->m           = m_;
    this->n           = n_;
    this->k           = k_;
    this->ldA         = lda_;
    this->ldB         = ldb_;
    this->ldC         = ldc_;
    this->strideA     = -1;
    this->strideB     = -1;
    this->strideC     = -1;
    this->alpha       = -1;
    this->beta        = -1;
    this->batch_count = -1;
    this->dataType    = dataType_;
}

GemmDesc::GemmDesc(bool isColMajor_,
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
    : isColMajor(isColMajor_),
      transA(transA_),
      transB(transB_),
      m(m_),
      n(n_),
      k(k_),
      ldA(lda_),
      ldB(ldb_),
      ldC(ldc_),
      strideA(strideA_),
      strideB(strideB_),
      strideC(strideC_),
      alpha(alpha_),
      beta(beta_),
      batch_count(batch_count_),
      dataType(dataType_)
{
}

bool GemmDesc::GetIsColMajor() const { return this->isColMajor; }

bool GemmDesc::GetTransA() const { return this->transA; }

bool GemmDesc::GetTransB() const { return this->transB; }

int GemmDesc::GetM() const { return this->m; }

int GemmDesc::GetN() const { return this->n; }

int GemmDesc::GetK() const { return this->k; }

int GemmDesc::GetldA() const { return this->ldA; }

int GemmDesc::GetldB() const { return this->ldB; }

int GemmDesc::GetldC() const { return this->ldC; }

long long int GemmDesc::GetStrideA() const { return this->strideA; }

long long int GemmDesc::GetStrideB() const { return this->strideB; }

long long int GemmDesc::GetStrideC() const { return this->strideC; }

double GemmDesc::GetAlpha() const { return this->alpha; }

double GemmDesc::GetBeta() const { return this->beta; }

int GemmDesc::GetBatchCount() const { return this->batch_count; }

miopenDataType_t GemmDesc::GetMIOpenDataType() const { return this->dataType; }

void GemmDesc::SetIsColMajor(bool icm) { this->isColMajor = icm; }

std::ostream& operator<<(std::ostream& stream, const GemmDesc& gemm_desc)
{
    return stream << "{"
                  << "isColMajor " << gemm_desc.isColMajor << ", "
                  << "transA " << gemm_desc.transA << ", "
                  << "transB " << gemm_desc.transB << ", "
                  << "m " << gemm_desc.m << ", "
                  << "n " << gemm_desc.n << ", "
                  << "k " << gemm_desc.k << ", "
                  << "ldA " << gemm_desc.ldA << ", "
                  << "ldB " << gemm_desc.ldB << ", "
                  << "ldC " << gemm_desc.ldC << ", "
                  << "batch_count " << gemm_desc.batch_count << ", "
                  << "strideA " << gemm_desc.strideA << ", "
                  << "strideB " << gemm_desc.strideB << ", "
                  << "strideC " << gemm_desc.strideC << ", "
                  << "alpha " << gemm_desc.alpha << ", "
                  << "beta " << gemm_desc.beta << ", "
                  << "dataType " << gemm_desc.dataType << "} ";
}

} // namespace miopen
