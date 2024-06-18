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

bool isBroadcastable(const TensorDescriptor& x, const TensorDescriptor& y);
template <int N>
tensor_view_t<N> broadcastTo(const tensor_view_t<N>& in, const tensor_view_t<N>& target);
extern template tensor_view_t<5> broadcastTo(const tensor_view_t<5>&, const tensor_view_t<5>&);
template <int N>
int64_t checkBroadcastedContiguous(const tensor_view_t<N>& tensorView);
extern template int64_t checkBroadcastedContiguous(const tensor_view_t<5>&);
bool isAllContiguous(const tensor_view_t<5>& inputGrad_tv,
                     const tensor_view_t<5>& otherGrad_tv,
                     const tensor_view_t<5>& cond_tv,
                     const tensor_view_t<5>& outputGrad_tv);
bool isAllBroadcastedContiguous(const tensor_view_t<5>& inputGrad_tv,
                                const tensor_view_t<5>& otherGrad_tv,
                                const tensor_view_t<5>& cond_tv,
                                const tensor_view_t<5>& outputGrad_tv);
bool isConditionBroadcasted(const tensor_view_t<5>& inputGrad_tv,
                            const tensor_view_t<5>& otherGrad_tv,
                            const tensor_view_t<5>& cond_tv);

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
        if(!isBroadcastable(inputGradDesc, outputGradDesc))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "WHERE::ProblemDescription: Dimensions of input gradient and output "
                         "gradient must be "
                         "broadcastable.");
        }

        if(!isBroadcastable(otherGradDesc, outputGradDesc))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "WHERE::ProblemDescription: Dimensions of other gradient and output "
                         "gradient must be "
                         "broadcastable.");
        }

        inputGrad_tv  = get_inner_expanded_tv<5>(inputGradDesc);
        otherGrad_tv  = get_inner_expanded_tv<5>(otherGradDesc);
        condition_tv  = get_inner_expanded_tv<5>(conditionDesc);
        outputGrad_tv = get_inner_expanded_tv<5>(outputGradDesc);

        inputGrad_tv = broadcastTo(inputGrad_tv, outputGrad_tv);
        otherGrad_tv = broadcastTo(otherGrad_tv, outputGrad_tv);
        condition_tv = broadcastTo(condition_tv, outputGrad_tv);
    }

    Direction GetDirection() const { return direction; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetConditionDesc() const { return conditionDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOtherGradDesc() const { return otherGradDesc; }
    const tensor_view_t<5>& GetOutputGrad_tv() const { return outputGrad_tv; }
    const tensor_view_t<5>& GetCondition_tv() const { return condition_tv; }
    const tensor_view_t<5>& GetInputGrad_tv() const { return inputGrad_tv; }
    const tensor_view_t<5>& GetOtherGrad_tv() const { return otherGrad_tv; }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            return false;
        }
        if(otherGradDesc.GetType() != outputGradDesc.GetType())
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

    tensor_view_t<5> outputGrad_tv;
    tensor_view_t<5> condition_tv;
    tensor_view_t<5> inputGrad_tv;
    tensor_view_t<5> otherGrad_tv;
};

} // namespace where

} // namespace miopen
