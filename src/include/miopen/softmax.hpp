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
#ifndef MIOPEN_SOFTMAX_HPP_
#define MIOPEN_SOFTMAX_HPP_

#include <miopen/common.hpp>
#include <miopen/miopen.h>
#include <miopen/object.hpp>

#include <nlohmann/json_fwd.hpp>

namespace miopen {

struct Handle;
struct TensorDescriptor;

struct MIOPEN_INTERNALS_EXPORT SoftmaxDescriptor : miopenSoftmaxDescriptor
{
    SoftmaxDescriptor() {}

    float GetAlpha() const { return alpha; }
    float GetBeta() const { return beta; }
    miopenSoftmaxAlgorithm_t GetAlgorithm() const { return algorithm; }
    miopenSoftmaxMode_t GetMode() const { return mode; }

    void SetParams(float alpha_,
                   float beta_,
                   miopenSoftmaxAlgorithm_t algorithm_,
                   miopenSoftmaxMode_t mode_)
    {
        alpha     = alpha_;
        beta      = beta_;
        algorithm = algorithm_;
        mode      = mode_;
    }

    MIOPEN_INTERNALS_EXPORT friend std::ostream& operator<<(std::ostream& stream,
                                                            const SoftmaxDescriptor& x);

    friend void to_json(nlohmann::json& json, const SoftmaxDescriptor& descriptor);
    friend void from_json(const nlohmann::json& json, SoftmaxDescriptor& descriptor);

private:
    float alpha;
    float beta;
    miopenSoftmaxAlgorithm_t algorithm;
    miopenSoftmaxMode_t mode;
};

MIOPEN_INTERNALS_EXPORT miopenStatus_t SoftmaxForward(const Handle& handle,
                                                      const void* alpha,
                                                      const void* beta,
                                                      const TensorDescriptor& xDesc,
                                                      ConstData_t x,
                                                      const TensorDescriptor& yDesc,
                                                      Data_t y,
                                                      miopenSoftmaxAlgorithm_t algorithm,
                                                      miopenSoftmaxMode_t mode,
                                                      int x_offset = 0,
                                                      int y_offset = 0);

MIOPEN_INTERNALS_EXPORT miopenStatus_t SoftmaxBackward(const Handle& handle,
                                                       const void* alpha,
                                                       const TensorDescriptor& yDesc,
                                                       ConstData_t y,
                                                       const TensorDescriptor& dyDesc,
                                                       ConstData_t dy,
                                                       const void* beta,
                                                       const TensorDescriptor& dxDesc,
                                                       Data_t dx,
                                                       miopenSoftmaxAlgorithm_t algorithm,
                                                       miopenSoftmaxMode_t mode,
                                                       int y_offset  = 0,
                                                       int dy_offset = 0,
                                                       int dx_offset = 0);

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenSoftmaxDescriptor, miopen::SoftmaxDescriptor);

#endif // _MIOPEN_SOFTMAX_HPP_
