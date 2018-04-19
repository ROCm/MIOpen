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
#include <miopen/activ.hpp>
#include <miopen/logger.hpp>

namespace miopen {

ActivationDescriptor::ActivationDescriptor() {}

ActivationDescriptor::ActivationDescriptor(miopenActivationMode_t m, const double* pparms)
    : parms(pparms, pparms + 3), mode(m)
{
}

ActivationDescriptor::ActivationDescriptor(miopenActivationMode_t m,
                                           double alpha,
                                           double beta,
                                           double gamma)
    : parms({alpha, beta, gamma}), mode(m)
{
}

miopenActivationMode_t ActivationDescriptor::GetMode() const { return this->mode; }

double ActivationDescriptor::GetAlpha() const { return this->parms[0]; }

double ActivationDescriptor::GetBeta() const { return this->parms[1]; }

double ActivationDescriptor::GetGamma() const { return this->parms[2]; }
std::ostream& operator<<(std::ostream& stream, const ActivationDescriptor& x)
{
    MIOPEN_LOG_ENUM(stream,
                    x.mode,
                    miopenActivationPASTHRU,
                    miopenActivationLOGISTIC,
                    miopenActivationTANH,
                    miopenActivationRELU,
                    miopenActivationSOFTRELU,
                    miopenActivationABS,
                    miopenActivationPOWER,
                    miopenActivationCLIPPEDRELU,
                    miopenActivationLEAKYRELU,
                    miopenActivationELU)
        << ", ";
    LogRange(stream, x.parms, ", ") << ", ";
    return stream;
}
} // namespace miopen
