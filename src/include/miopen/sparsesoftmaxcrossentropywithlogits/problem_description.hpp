/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/miopen.h>
#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace sparsesoftmaxcrossentropywithlogits {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& targetDesc_,
                          const TensorDescriptor& outputDesc_,
                          const TensorDescriptor& backpropDesc_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          outputDesc(outputDesc_),
          backpropDesc(backpropDesc_)
    {
        IsSameType();
        IsValidType();
        IsValidDims();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetBackpropDesc() const { return backpropDesc; }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType() ||
           inputDesc.GetType() != backpropDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SparseSoftmaxCrossEntropyWithLogitsForward: Data types do not match.");
        return true;
    }

    bool IsValidType() const
    {
        if(targetDesc.GetType() != miopenInt32 && targetDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SparseSoftmaxCrossEntropyWithLogitsForward: target tensor must be int32 "
                         "or int64.");
        }
        return true;
    }

    bool IsValidDims() const
    {
        if(inputDesc.GetNumDims() != 2 || targetDesc.GetNumDims() != 1 ||
           outputDesc.GetNumDims() != 1 || backpropDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SparseSoftmaxCrossEntropyWithLogitsForward: Tensor sizes do not match.");
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        return inputDesc.IsContiguous() && targetDesc.IsContiguous() && outputDesc.IsContiguous() &&
               backpropDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor backpropDesc;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& backpropDesc_,
                          const TensorDescriptor& inputGradDesc_)
        : outputGradDesc(outputGradDesc_),
          backpropDesc(backpropDesc_),
          inputGradDesc(inputGradDesc_)
    {
        IsSameType();
        IsValidDims();
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType() ||
           inputGradDesc.GetType() != backpropDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "SparseSoftmaxCrossEntropyWithLogitsBackward: Data types do not match.");
        return true;
    }

    bool IsValidDims() const
    {
        if(inputGradDesc.GetNumDims() != 2 || outputGradDesc.GetNumDims() != 1 ||
           backpropDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SparseSoftmaxCrossEntropyWithLogitsBackward: Tensor sizes do not match.");
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        return outputGradDesc.IsContiguous() && backpropDesc.IsContiguous() &&
               inputGradDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor outputGradDesc;
    TensorDescriptor backpropDesc;
    TensorDescriptor inputGradDesc;
};

} // namespace sparsesoftmaxcrossentropywithlogits

} // namespace miopen
