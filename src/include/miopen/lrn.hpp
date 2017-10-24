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
#ifndef MIOPEN_LRN_HPP_
#define MIOPEN_LRN_HPP_

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <vector>

namespace miopen {

struct LRNDescriptor : miopenLRNDescriptor
{
    LRNDescriptor();
    LRNDescriptor(miopenLRNMode_t m, unsigned int pn, const double* pparms);
    LRNDescriptor(miopenLRNMode_t m, unsigned int pn, std::vector<double> pparms);

    miopenLRNMode_t GetMode() const;
    unsigned int GetN() const;
    double GetAlpha() const;
    double GetBeta() const;
    double GetK() const;

    miopenStatus_t Forward(Handle& handle,
                           const void* alpha,
                           const TensorDescriptor& xDesc,
                           ConstData_t x,
                           const void* beta,
                           const TensorDescriptor& yDesc,
                           Data_t y,
                           bool do_backward,
                           Data_t workSpace);

    miopenStatus_t Backward(Handle& handle,
                            const void* alpha,
                            const TensorDescriptor& yDesc,
                            ConstData_t y,
                            const TensorDescriptor& dyDesc,
                            ConstData_t dy,
                            const TensorDescriptor& xDesc,
                            ConstData_t x,
                            const void* beta,
                            const TensorDescriptor& dxDesc,
                            Data_t dx,
                            ConstData_t workSpace);

    friend std::ostream& operator<<(std::ostream& stream, const LRNDescriptor& x);

    private:
    unsigned int lrnN = 0;
    std::vector<double> parms;

    miopenLRNMode_t mode = miopenLRNWithinChannel;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenLRNDescriptor, miopen::LRNDescriptor);
#endif // _MIOPEN_LRN_HPP_
