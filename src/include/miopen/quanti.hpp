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
#ifndef MIOPEN_QUANTI_HPP_
#define MIOPEN_QUANTI_HPP_

#include <miopen/common.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>
#include <vector>

namespace miopen {

struct Handle;
struct TensorDescriptor;

struct QuantizationDescriptor : miopenQuantizationDescriptor
{
    QuantizationDescriptor();
    QuantizationDescriptor(double scaler, double bias);

    void SetScaler();
    void SetBias();
    double GetScaler() const;
    double GetBias() const;

    miopenStatus_t Quantize(Handle& handle,
                           const TensorDescriptor& inDesc,
                           ConstData_t in,
                           const void* scaler,                           
                           const void* bias,
                           const TensorDescriptor& outDesc,
                           Data_t out);

    friend std::ostream& operator<<(std::ostream& stream, const QuantizationDescriptor& x);

private:

};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenQuantizationDescriptor, miopen::QuantizationDescriptor);
#endif // _MIOPEN_QUANTI_HPP_
