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

#include <cstddef>
#include <cstdint>

#include "miopen/errors.hpp"
#include "miopen/tensor_view_utils.hpp"
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <miopen/activ.hpp>

namespace miopen {

struct NetworkConfig;

namespace where {

bool isSameShape(const TensorDescriptor& x, const TensorDescriptor& y);

template <int N>
int64_t checkBroadcastedContiguous(const tensor_view_t<N>& tensorView);
extern template int64_t checkBroadcastedContiguous(const tensor_view_t<5>&);

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
        // TODO: support broadcastable ops
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

        inputGrad_tv  = get_inner_expanded_tv<5>(inputGradDesc);
        otherGrad_tv  = get_inner_expanded_tv<5>(otherGradDesc);
        condition_tv  = get_inner_expanded_tv<5>(conditionDesc);
        outputGrad_tv = get_inner_expanded_tv<5>(outputGradDesc);
    }

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
        if(!inputGradDesc.GetLengths().empty() &&
           inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            return false;
        }
        if(!otherGradDesc.GetLengths().empty() &&
           otherGradDesc.GetType() != outputGradDesc.GetType())
        {
            return false;
        }

        return true;
    }

    bool isAllSameShape() const
    {
        // TODO: inputGradDesc and otherGradDesc not exist
        return isSameShape(outputGradDesc, inputGradDesc) &&
               isSameShape(outputGradDesc, otherGradDesc) &&
               isSameShape(outputGradDesc, conditionDesc);
    }

    bool isAllContiguous() const
    {
        // TODO: inputGrad and otherGrad not exist
        return outputGradDesc.IsContiguous() && conditionDesc.IsContiguous() &&
               inputGradDesc.IsContiguous() && otherGradDesc.IsContiguous();
    }

    bool isAllBroadcastedContiguous() const
    {
        auto cond_contig_size       = checkBroadcastedContiguous(condition_tv);
        auto input_grad_contig_size = checkBroadcastedContiguous(inputGrad_tv);
        auto other_grad_contig_size = checkBroadcastedContiguous(otherGrad_tv);

        bool is_all_broadcasted_contiguous =
            (cond_contig_size > 0) && (input_grad_contig_size > 0) &&
            (other_grad_contig_size > 0) && outputGradDesc.IsContiguous();
        return is_all_broadcasted_contiguous;
    }

    bool isConditionBroadcasted() const
    {
        auto cond_contig_size       = checkBroadcastedContiguous(condition_tv);
        auto input_grad_contig_size = checkBroadcastedContiguous(inputGrad_tv);
        auto other_grad_contig_size = checkBroadcastedContiguous(otherGrad_tv);

        bool is_condition_broadcasted = (cond_contig_size > 0) &&
                                        ((input_grad_contig_size % cond_contig_size == 0) ||
                                         (other_grad_contig_size % cond_contig_size == 0)) &&
                                        cond_contig_size >= static_cast<int64_t>(256 * 120 * 4);
        return is_condition_broadcasted;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
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
