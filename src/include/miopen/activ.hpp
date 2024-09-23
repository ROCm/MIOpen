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
#ifndef MIOPEN_ACTIV_HPP_
#define MIOPEN_ACTIV_HPP_

#include <miopen/common.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>

#include <nlohmann/json_fwd.hpp>

#include <vector>

namespace miopen {

struct Handle;
struct TensorDescriptor;

struct MIOPEN_INTERNALS_EXPORT ActivationDescriptor : miopenActivationDescriptor
{
    ActivationDescriptor();
    ActivationDescriptor(miopenActivationMode_t m, const double* pparms);
    ActivationDescriptor(miopenActivationMode_t m, double alpha, double beta, double gamma);

    miopenActivationMode_t GetMode() const;
    double GetAlpha() const;
    double GetBeta() const;
    double GetGamma() const;

    miopenStatus_t Forward(Handle& handle,
                           const void* alpha,
                           const TensorDescriptor& xDesc,
                           ConstData_t x,
                           const void* beta,
                           const TensorDescriptor& yDesc,
                           Data_t y,
                           size_t xOffset = 0,
                           size_t yOffset = 0) const;

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
                            size_t yOffset  = 0,
                            size_t dyOffset = 0,
                            size_t xOffset  = 0,
                            size_t dxOffset = 0) const;

    friend std::ostream& operator<<(std::ostream& stream, const ActivationDescriptor& x);

    friend void to_json(nlohmann::json& json, const ActivationDescriptor& descriptor);
    friend void from_json(const nlohmann::json& json, ActivationDescriptor& descriptor);

private:
    std::vector<double> parms;

    miopenActivationMode_t mode = miopenActivationPASTHRU;
};

} // namespace miopen
MIOPEN_DEFINE_OBJECT(miopenActivationDescriptor, miopen::ActivationDescriptor);
#endif // _MIOPEN_ACTIV_HPP_
