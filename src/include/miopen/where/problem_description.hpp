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
        if(inputDesc.GetLengths().size() != outputDesc.GetLengths().size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Where::ProblemDescription: Number of tensor dimension do not match.");
        }

        for(int32_t i = 0; i < inputDesc.GetLengths().size(); i++)
        {
            if(inputDesc.GetLengths()[i] != outputDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Where::ProblemDescription: Dimension sizes don't match between "
                             "input tensor and output tensor.");
            }
        }
    }

    Direction GetDirection() const { return direction; }
    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }

    bool IsSameType() const
    {
        if(direction == Direction::Forward)
        {
            if(inputDesc.GetType() != outputDesc.GetType())
            {
                return false;
            }
        }
        else
        {
            if(inputDesc.GetType() != inputGradDesc.GetType() ||
               inputGradDesc.GetType() != outputGradDesc.GetType())
            {
                return false;
            }
        }

        return true;
    }

    bool IsAllPacked() const
    {
        if(direction == Direction::Forward)
        {
            if(!(inputDesc.IsPacked() && outputDesc.IsPacked()))
            {
                return false;
            }
        }
        else
        {
            if(!(inputDesc.IsPacked() && inputGradDesc.IsPacked() && outputGradDesc.IsPacked()))
            {
                return false;
            }
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    Direction direction;
    TensorDescriptor inputDesc;
    TensorDescriptor inputGradDesc;
    TensorDescriptor otherDesc;
    TensorDescriptor otherGradDesc;
    TensorDescriptor conditionDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor outputGradDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
    NetworkConfig MakeBackwardNetworkConfig() const;
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
        if(inputGradDesc.GetLengths().size() != otherGradDesc.GetLengths().size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Where::ProblemDescription: Number of tensor dimension do not match.");
        }

        for(int32_t i = 0; i < inputGradDesc.GetLengths().size(); i++)
        {
            if(inputGradDesc.GetLengths()[i] != otherGradDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "Where::ProblemDescription: Dimension sizes don't match between "
                             "input tensor and output tensor.");
            }
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
        if(!(inputGradDesc.IsPacked() && otherGradDesc.IsPacked() && conditionDesc.IsPacked()))
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

    NetworkConfig MakeForwardNetworkConfig() const;
    NetworkConfig MakeBackwardNetworkConfig() const;
};

} // namespace where

} // namespace miopen
