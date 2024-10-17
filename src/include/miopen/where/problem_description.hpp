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

#include <miopen/errors.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace where {

bool isSameShape(const TensorDescriptor& x, const TensorDescriptor& y);

struct BackwardProblemDescription : ProblemDescriptionBase
{
    // Backward constructor
    BackwardProblemDescription(const TensorDescriptor& outputGradDesc_,
                               const TensorDescriptor& conditionDesc_,
                               const TensorDescriptor& inputGradDesc_,
                               const TensorDescriptor& otherGradDesc_)
        : outputGradDesc(outputGradDesc_),
          conditionDesc(conditionDesc_),
          inputGradDesc(inputGradDesc_),
          otherGradDesc(otherGradDesc_)
    {
        if(!isAllSameShape())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "WHERE::ProblemDescription: All tensors must have the same shape.");
        }

        if(!isAllContiguous())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "WHERE::ProblemDescription: All tensors must be contiguous.");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "WHERE::ProblemDescription: All tensors must have the same type.");
        }

        isInputGradRequired = !inputGradDesc.GetLengths().empty();
        isOtherGradRequired = !otherGradDesc.GetLengths().empty();
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetConditionDesc() const { return conditionDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOtherGradDesc() const { return otherGradDesc; }

    bool IsSameType() const
    {
        if(isInputGradRequired && inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            return false;
        }
        if(isOtherGradRequired && otherGradDesc.GetType() != outputGradDesc.GetType())
        {
            return false;
        }

        return true;
    }

    bool isAllSameShape() const
    {
        if(isInputGradRequired && !isSameShape(inputGradDesc, outputGradDesc))
        {
            return false;
        }
        if(isOtherGradRequired && !isSameShape(otherGradDesc, outputGradDesc))
        {
            return false;
        }
        return isSameShape(outputGradDesc, conditionDesc);
    }

    bool isAllContiguous() const
    {
        if(isInputGradRequired && !inputGradDesc.IsContiguous())
        {
            return false;
        }
        if(isOtherGradRequired && !otherGradDesc.IsContiguous())
        {
            return false;
        }
        return outputGradDesc.IsContiguous() && conditionDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor outputGradDesc;
    TensorDescriptor conditionDesc;
    TensorDescriptor inputGradDesc;
    TensorDescriptor otherGradDesc;

    bool isInputGradRequired;
    bool isOtherGradRequired;
};

} // namespace where

} // namespace miopen
