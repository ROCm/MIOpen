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

#include "miopen/tensor_view_utils.hpp"
#include <cstdint>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>

#include <string>

namespace miopen {

struct NetworkConfig;

namespace where {

enum class Direction
{
    Forward,
    Backward,
};

struct ForwardProblemDescription : ProblemDescriptionBase
{
    // Forward constructor
    ForwardProblemDescription(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& otherDesc_,
                              const TensorDescriptor& conditionDesc_,
                              const TensorDescriptor& outputDesc_)
        : direction(Direction::Forward),
          inputDesc(inputDesc_),
          otherDesc(otherDesc_),
          conditionDesc(conditionDesc_),
          outputDesc(outputDesc_)
    {
        if(!(isBroadcastable(inputDesc, otherDesc) && isBroadcastable(otherDesc, conditionDesc))) {
            MIOPEN_THROW(miopenStatusBadParm,
                         "WHERE::ProblemDescription: Dimensions of input, other, condition must be broadcastable.");            
        }
    }

    Direction GetDirection() const { return direction; }
    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOtherDesc() const { return otherDesc; }
    const TensorDescriptor& GetConditionDesc() const { return conditionDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != otherDesc.GetType() || otherDesc.GetType() != outputDesc.GetType())
        {
            return false;
        }

        return true;
    }

    bool IsAllPacked() const
    {
        if(!(inputDesc.IsPacked() && otherDesc.IsPacked() && outputDesc.IsPacked()))
        {
            return false;
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    Direction direction;
    TensorDescriptor inputDesc;
    TensorDescriptor otherDesc;
    TensorDescriptor conditionDesc;
    TensorDescriptor outputDesc;
};

struct BackwardProblemDescription : ProblemDescriptionBase
{
    // Backward constructor
    BackwardProblemDescription(const TensorDescriptor& outputGradDesc_,
                               const TensorDescriptor& conditionDesc_,
                               const TensorDescriptor& inputGradDesc_,
                               const TensorDescriptor& otherGradDesc_)
        : direction(Direction::Backward),
          outputGradDesc(outputGradDesc_),
          conditionDesc(conditionDesc_),
          inputGradDesc(inputGradDesc_),
          otherGradDesc(otherGradDesc_)
    {
        if(!(isBroadcastable(inputGradDesc, otherGradDesc) && isBroadcastable(otherGradDesc, conditionDesc))) {
            MIOPEN_THROW(miopenStatusBadParm,
                         "WHERE::ProblemDescription: Dimensions of input, other, condition must be broadcastable.");
        }
    }

    Direction GetDirection() const { return direction; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetConditionDesc() const { return conditionDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOtherGradDesc() const { return otherGradDesc; }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != otherGradDesc.GetType() ||
           inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            return false;
        }

        return true;
    }

    bool IsAllPacked() const
    {
        if(!(inputGradDesc.IsPacked() && otherGradDesc.IsPacked() && outputGradDesc.IsPacked()))
        {
            return false;
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    Direction direction;
    TensorDescriptor outputGradDesc;
    TensorDescriptor conditionDesc;
    TensorDescriptor inputGradDesc;
    TensorDescriptor otherGradDesc;
};

} // namespace where

} // namespace miopen
