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
        : algorithm(algorithm_),
          mode(mode_),

          xdxDesc(xDesc_),
          x(x_),
          dx(nullptr),

          yDesc(yDesc_),
          forward_y(y_),
          backward_y(nullptr),

          dy(nullptr),

          xdx_offset(x_offset_),
          y_offset(y_offset_),
          dy_offset(0)
    {
        InitializeAlphaBeta(alpha_, beta_);
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
                 int y_offset_  = 0,
                 int dy_offset_ = 0,
                 int dx_offset_ = 0)
        : algorithm(algorithm_),
          mode(mode_),

          xdxDesc(dxDesc_),
          x(nullptr),
          dx(dx_),

          yDesc(yDesc_),
          forward_y(nullptr),
          backward_y(y_),

          dyDesc(dyDesc_),
          dy(dy_),

          xdx_offset(dx_offset_),
          y_offset(y_offset_),
          dy_offset(dy_offset_)
    {
        InitializeAlphaBeta(alpha_, beta_);
    }

    std::size_t GetWorkspaceSize() const { return 0; }
    Data_t GetWorkspace() const { return nullptr; }

public:
    float alpha;
    float beta;
    miopenSoftmaxAlgorithm_t algorithm;
    miopenSoftmaxMode_t mode;

    // xdxDesc is used for both forward and backward
    TensorDescriptor xdxDesc;
    ConstData_t x;
    Data_t dx;

    TensorDescriptor yDesc;
    Data_t forward_y;
    ConstData_t backward_y;

    // backward specific part
    TensorDescriptor dyDesc;
    ConstData_t dy;

    // xdx_offset is used for both forward and backward
    int xdx_offset;
    int y_offset;
    int dy_offset;

private:
    void InitializeAlphaBeta(const void* alpha_, const void* beta_)
    {
        alpha = 0.0f;
        beta  = 0.0f;

        if(alpha_ != nullptr)
        {
            alpha = *(static_cast<const float*>(alpha_));
        }

        if(beta_ != nullptr)
        {
            beta = *(static_cast<const float*>(beta_));
        }
    }
};

} // namespace softmax
} // namespace miopen
