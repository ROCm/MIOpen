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
#include <miopen/logger.hpp>
#include <miopen/lrn.hpp>

namespace miopen {

LRNDescriptor::LRNDescriptor() {}

LRNDescriptor::LRNDescriptor(miopenLRNMode_t m, const unsigned int pn, const double* pparms)
    : lrnN(pn), parms(pparms, pparms + 3), mode(m)
{
}

LRNDescriptor::LRNDescriptor(miopenLRNMode_t m, unsigned int pn, std::vector<double> pparms)
    : lrnN(pn), parms(std::move(pparms)), mode(m)
{
}
miopenLRNMode_t LRNDescriptor::GetMode() const { return this->mode; }

unsigned int LRNDescriptor::GetN() const { return this->lrnN; }

double LRNDescriptor::GetAlpha() const { return this->parms[0]; }

double LRNDescriptor::GetBeta() const { return this->parms[1]; }

double LRNDescriptor::GetK() const { return this->parms[2]; }
std::ostream& operator<<(std::ostream& stream, const LRNDescriptor& x)
{
    MIOPEN_LOG_ENUM(stream, x.mode, miopenLRNWithinChannel, miopenLRNCrossChannel) << ", ";
    stream << x.lrnN << ", ";
    LogRange(stream, x.parms, ", ") << ", ";
    return stream;
}
} // namespace miopen
