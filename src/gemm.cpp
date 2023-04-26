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
#include <miopen/tensor.hpp>
#include <miopen/gemm.hpp>
#include <miopen/logger.hpp>

namespace miopen {

GemmDesc::GemmDesc() {}

GemmDesc::GemmDesc(long long int lda_, long long int ldb_, long long int ldc_)
{
    this->isColMajor  = false;
    this->transA      = false;
    this->transB      = false;
    this->ldA         = lda_;
    this->ldB         = ldb_;
    this->ldC         = ldc_;
    this->strideA     = -1;
    this->strideB     = -1;
    this->strideC     = -1;
    this->batch_count = -1;
}

GemmDesc::GemmDesc(bool isColMajor_,
                   bool transA_,
                   bool transB_,
                   int lda_,
                   int ldb_,
                   int ldc_,
                   long long int strideA_,
                   long long int strideB_,
                   long long int strideC_,
                   int batch_count_)
    : isColMajor(isColMajor_),
      transA(transA_),
      transB(transB_),
      ldA(lda_),
      ldB(ldb_),
      ldC(ldc_),
      strideA(strideA_),
      strideB(strideB_),
      strideC(strideC_),
      batch_count(batch_count_)
{
}

bool GemmDesc::GetIsColMajor() const { return this->isColMajor; }

bool GemmDesc::GetTransA() const { return this->transA; }

bool GemmDesc::GetTransB() const { return this->transB; }

int GemmDesc::GetldA() const { return this->ldA; }

int GemmDesc::GetldB() const { return this->ldB; }

int GemmDesc::GetldC() const { return this->ldC; }

long long int GemmDesc::GetStrideA() const { return this->strideA; }

long long int GemmDesc::GetStrideB() const { return this->strideB; }

long long int GemmDesc::GetStrideC() const { return this->strideC; }

int GemmDesc::GetBatchCount() const { return this->batch_count; }

void GemmDesc::SetIsColMajor(bool icm) { this->isColMajor = icm; }

std::ostream& operator<<(std::ostream& stream, const GemmDesc& gemm_desc)
{
    return stream << "{"
                  << "isColMajor " << gemm_desc.isColMajor << ", "
                  << "transA " << gemm_desc.transA << ", "
                  << "transB " << gemm_desc.transB << ", "
                  << "ldA " << gemm_desc.ldA << ", "
                  << "ldB " << gemm_desc.ldB << ", "
                  << "ldC " << gemm_desc.ldC << ", "
                  << "batch_count " << gemm_desc.batch_count << ", "
                  << "strideA " << gemm_desc.strideA << ", "
                  << "strideB " << gemm_desc.strideB << ", "
                  << "strideC " << gemm_desc.strideC << "} ";
}

TensorDescriptor GemmDesc::GetOutputTensor(const TensorDescriptor& ADesc,
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

    return {ADesc.GetType(), {a_lens[0], b_lens[1]}};
}

} // namespace miopen
