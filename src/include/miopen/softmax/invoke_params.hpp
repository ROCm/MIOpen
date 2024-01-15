/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#pragma once

#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

namespace miopen {
namespace softmax {

struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams(const void* alpha_,
                 const void* beta_,
                 const TensorDescriptor& xDesc_,
                 ConstData_t x_,
                 const TensorDescriptor& yDesc_,
                 Data_t y_,
                 miopenSoftmaxAlgorithm_t algorithm_,
                 miopenSoftmaxMode_t mode_,
                 int x_offset_ = 0,
                 int y_offset_ = 0)
        : alpha(alpha_),
          beta(beta_),
          xDesc(xDesc_),
          x(x_),
          yDesc(yDesc_),
          y(y_),
          algorithm(algorithm_),
          mode(mode_),
          x_offset(x_offset_),
          y_offset(y_offset_)
    {
    }

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }

    // miopenHandle_t handle;
    const void* alpha;
    const void* beta;
    const TensorDescriptor& xDesc;
    ConstData_t x;
    const TensorDescriptor& yDesc;
    Data_t y;
    miopenSoftmaxAlgorithm_t algorithm;
    miopenSoftmaxMode_t mode;
    int x_offset = 0;
    int y_offset = 0;
};

} // namespace softmax
} // namespace miopen
