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
          algorithm(algorithm_),
          mode(mode_),

          xdxDesc(xDesc_),
          x(x_),
          dx(nullptr),

          yDesc(yDesc_),
          forward_y(y_),
          backward_y(nullptr),

          dyDesc(yDesc_),
          dy(nullptr),

          xdx_offset(x_offset_),
          y_offset(y_offset_),
          dy_offset(0)
    {
    }

    InvokeParams(const void* alpha_,
                 const void* beta_,
                 const TensorDescriptor& yDesc_,
                 ConstData_t y_,
                 const TensorDescriptor& dyDesc_,
                 ConstData_t dy_,
                 const TensorDescriptor& dxDesc_,
                 Data_t dx_,
                 miopenSoftmaxAlgorithm_t algorithm_,
                 miopenSoftmaxMode_t mode_,
                 int y_offset_,
                 int dy_offset_,
                 int dx_offset_)
        : alpha(alpha_),
          beta(beta_),
          algorithm(algorithm_),
          mode(mode_),

          xdxDesc(dxDesc_),
          x(nullptr),
          dx(dx_),

          yDesc(yDesc_),
          forward_y(nullptr),
          backward_y(y_),

          dyDesc(dyDesc_),
          dy(dy_),

          xdx_offset(0),
          y_offset(y_offset_),
          dy_offset(dy_offset_)
    {
    }

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }

public:
    const void* alpha;
    const void* beta;
    miopenSoftmaxAlgorithm_t algorithm;
    miopenSoftmaxMode_t mode;

    // xdxDesc is used for both forward and bacward
    const TensorDescriptor& xdxDesc;
    ConstData_t x;
    Data_t dx;

    const TensorDescriptor& yDesc;
    Data_t forward_y;
    ConstData_t backward_y;

    // backward specific part
    const TensorDescriptor& dyDesc;
    ConstData_t dy;

    // xdx_offset is used for both forward and bacward
    int xdx_offset;
    int y_offset;
    int dy_offset;
};

} // namespace softmax
} // namespace miopen
