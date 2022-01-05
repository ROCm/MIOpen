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

GemmNewDescriptor::GemmNewDescriptor(//miopenGemmMode_t mode_,
                   bool isColMajor_,
                   bool transA_,
                   bool transB_,
                   int m_,
                   int n_, 
                   int k_,
                   double alpha_,
                   double beta_,
                   miopenDataType_t dataType_)
    : //mode(mode_),
      isColMajor(isColMajor_),
      transA(transA_),
      transB(transB_),
      m(m_),
      n(n_),
      k(k_),
      alpha(alpha_),
      beta(beta_),
      dataType(dataType_)
{
}

//miopenGemmMode_t GemmNewDescriptor::GetMode() const { return this->mode; }

bool GemmNewDescriptor::GetIsColMajor() const { return this->isColMajor; }

bool GemmNewDescriptor::GetTransA() const { return this->transA; }

bool GemmNewDescriptor::GetTransB() const { return this->transB; }

int GemmNewDescriptor::GetM() const { return this->m; }

int GemmNewDescriptor::GetN() const { return this->n; }

int GemmNewDescriptor::GetK() const { return this->k; }

double GemmNewDescriptor::GetAlpha() const { return this->alpha; }

double GemmNewDescriptor::GetBeta() const { return this->beta; }

miopenDataType_t GemmNewDescriptor::GetMIOpenDataType() const { return this->dataType; }

std::ostream& operator<<(std::ostream& stream, const GemmNewDescriptor& x)
{
    /*
    MIOPEN_LOG_ENUM(stream,
                    x.mode,
                    miopenGemmPASTHRU,
                    miopenGemmLOGISTIC,
                    miopenGemmTANH,
                    miopenGemmRELU,
                    miopenGemmSOFTRELU,
                    miopenGemmABS,
                    miopenGemmPOWER,
                    miopenGemmCLIPPEDRELU,
                    miopenGemmLEAKYRELU,
                    miopenGemmELU)
        << ", ";
    LogRange(stream, x.parms, ", ") << ", ";*/
    return stream;
}
} // namespace miopen
