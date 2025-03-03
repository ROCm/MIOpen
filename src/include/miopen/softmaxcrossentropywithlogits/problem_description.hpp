/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace softmaxcrossentropywithlogits {

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
        IsValidLength();
        IsSameType();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    size_t GetBatchSize() const { return inputDesc.GetLengths()[0]; }

    bool IsValidLength() const
    {
        if(inputDesc.GetNumDims() != 2 || targetDesc.GetNumDims() != 2 ||
           backpropDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Input, target, and backprop tensors size "
                         "!= 2 is not valid.");
        }

        if(outputDesc.GetNumDims() != 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Output tensor size != 1 is not valid.");
        }
        if(inputDesc.GetLengths()[0] != outputDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Tensor sizes do not match.");
        }
        for(size_t i = 0; i < inputDesc.GetNumDims(); ++i)
        {
            if(inputDesc.GetLengths()[i] != targetDesc.GetLengths()[i] ||
               inputDesc.GetLengths()[i] != backpropDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "SoftmaxCrossEntropyWithLogits: Tensor sizes do not match.");
            }
        }
        return true;
    }

    bool IsSameType() const
    {
        if(!(inputDesc.GetType() == targetDesc.GetType() &&
             inputDesc.GetType() == outputDesc.GetType() &&
             inputDesc.GetType() == backpropDesc.GetType()))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Tensor types do not match.");
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        return inputDesc.IsContiguous() && targetDesc.IsContiguous() && outputDesc.IsContiguous() &&
               backpropDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor backpropDesc;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& backpropDesc_,
                          const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& inputGradDesc_,
                          const TensorDescriptor& targetGradDesc_)
        : outputGradDesc(outputGradDesc_),
          backpropDesc(backpropDesc_),
          inputDesc(inputDesc_),
          inputGradDesc(inputGradDesc_),
          targetGradDesc(targetGradDesc_)
    {
        IsValidLength();
        IsSameType();
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }

    size_t GetBatchSize() const { return inputDesc.GetLengths()[0]; }

    bool IsValidLength() const
    {
        if(backpropDesc.GetNumDims() != 2 || inputDesc.GetNumDims() != 2 ||
           inputGradDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Input tensor size != 2 is not valid.");
        }
        if(targetGradDesc.GetNumDims() != 0 && targetGradDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Target Grad tensor sizes is not valid.");
        }
        if(outputGradDesc.GetNumDims() != 1)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "SoftmaxCrossEntropyWithLogits: Output grad tensor size != 1 is not valid.");
        }
        if(inputDesc.GetLengths()[0] != outputGradDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Tensor sizes do not match.");
        }
        for(size_t i = 0; i < inputDesc.GetNumDims(); ++i)
        {
            if(inputDesc.GetLengths()[i] != backpropDesc.GetLengths()[i] ||
               inputDesc.GetLengths()[i] != inputGradDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "SoftmaxCrossEntropyWithLogits: Tensor sizes do not match.");
            }
        }
        if(targetGradDesc.GetNumDims() == 2)
        {
            if(inputDesc.GetLengths()[0] != targetGradDesc.GetLengths()[0] ||
               inputDesc.GetLengths()[1] != targetGradDesc.GetLengths()[1])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "SoftmaxCrossEntropyWithLogits: Tensor sizes do not match.");
            }
        }
        return true;
    }

    bool IsSameType() const
    {
        if(!(inputDesc.GetType() == inputGradDesc.GetType() &&
             inputDesc.GetType() == targetGradDesc.GetType() &&
             inputDesc.GetType() == outputGradDesc.GetType() &&
             inputDesc.GetType() == backpropDesc.GetType()))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftmaxCrossEntropyWithLogits: Tensor types do not match.");
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        return outputGradDesc.IsContiguous() && backpropDesc.IsContiguous() &&
               inputDesc.IsContiguous() && inputGradDesc.IsContiguous() &&
               targetGradDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor outputGradDesc;
    TensorDescriptor backpropDesc;
    TensorDescriptor inputDesc;
    TensorDescriptor inputGradDesc;
    TensorDescriptor targetGradDesc;
};

} // namespace softmaxcrossentropywithlogits

} // namespace miopen
