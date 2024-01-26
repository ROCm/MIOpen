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

#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace softmax {

struct ProblemDescription : ProblemDescriptionBase
{
    // softmax forward constructor
    ProblemDescription(const void* alpha_,
                       const void* beta_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& yDesc_,
                       miopenSoftmaxAlgorithm_t algorithm_,
                       miopenSoftmaxMode_t mode_)
        : isForward(true),
          xdxDesc(xDesc_),
          yDesc(yDesc_),

          algorithm(algorithm_),
          mode(mode_)
    {
        CheckAndAssignAlphaBeta(alpha_, beta_);

        if(xdxDesc.GetType() != yDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(xdxDesc.GetLengths() != yDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }
    }

    ProblemDescription(const void* alpha_,
                       const void* beta_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& dyDesc_,
                       const TensorDescriptor& dxDesc_,
                       miopenSoftmaxAlgorithm_t algorithm_,
                       miopenSoftmaxMode_t mode_)
        : isForward(false),
          xdxDesc(dxDesc_),
          yDesc(yDesc_),
          dyDesc(dyDesc_),
          algorithm(algorithm_),
          mode(mode_)
    {
        CheckAndAssignAlphaBeta(alpha_, beta_);

        if(yDesc != dyDesc)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }

        if(xdxDesc.GetType() != dyDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(xdxDesc.GetLengths() != dyDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }
    }

    bool IsForward() const { return isForward; }
    miopenSoftmaxAlgorithm_t GetAlgorithm() const { return algorithm; }
    miopenSoftmaxMode_t GetMode() const { return mode; }
    float GetAlpha() const { return alpha; }
    float GetBeta() const { return beta; }

    // for forward
    const TensorDescriptor& GetXDesc() const { return xdxDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }

    // for backward
    const TensorDescriptor& GetdYDesc() const { return dyDesc; }
    const TensorDescriptor& GetdXDesc() const { return xdxDesc; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    void CheckAndAssignAlphaBeta(const void* alpha_, const void* beta_)
    {
        if(alpha_ == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Alpha value is nullptr");
        }

        if(beta_ == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Beta value is nullptr");
        }

        alpha = *(static_cast<const float*>(alpha_));
        beta  = *(static_cast<const float*>(beta_));
    }

    const bool isForward;

    float alpha;
    float beta;

    // for forward xDesc is stored in xdxDesc, for backward dxDesc is stored in xdxDesc
    TensorDescriptor xdxDesc;
    TensorDescriptor yDesc;
    TensorDescriptor dyDesc;

    const miopenSoftmaxAlgorithm_t algorithm;
    const miopenSoftmaxMode_t mode;
};

} // namespace softmax
} // namespace miopen
