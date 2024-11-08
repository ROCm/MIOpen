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

#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace marginrankingloss {

struct ProblemDescriptionForward : ProblemDescriptionBase
{
    ProblemDescriptionForward(const TensorDescriptor& input1Desc_,
                              const TensorDescriptor& input2Desc_,
                              const TensorDescriptor& targetDesc_,
                              const TensorDescriptor& outputDesc_,
                              float margin_,
                              miopenMarginRakningLossReductionMode_t reduction_mode_)
        : input1Desc(input1Desc_),
          input2Desc(input2Desc_),
          targetDesc(targetDesc_),
          outputDesc(outputDesc_),
          margin(margin_),
          reduction_mode(reduction_mode_)
    {
        IsSameLength();
        IsSameType();
        IsAllApplicableDims();
    }

    const TensorDescriptor& GetInput1Desc() const { return input1Desc; }
    const TensorDescriptor& GetInput2Desc() const { return input2Desc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const miopenMarginRakningLossReductionMode_t& GetReductionMode() const
    {
        return reduction_mode;
    }
    const float& GetMargin() const { return margin; }

    bool IsSameLength() const
    {
        auto input1Lengths = input1Desc.GetLengths();
        auto input2Lengths = input2Desc.GetLengths();
        auto targetLengths = targetDesc.GetLengths();
        auto outputLengths = outputDesc.GetLengths();
        if((input1Lengths != input2Lengths) || (input2Lengths != targetLengths) ||
           (targetLengths != outputLengths))
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "MarginRankingLossForward: All input tensors should have the same length.");
        }
        return true;
    }

    bool IsSameType() const
    {
        if((input1Desc.GetType() != input2Desc.GetType()) ||
           (input2Desc.GetType() != targetDesc.GetType()) ||
           (targetDesc.GetType() != outputDesc.GetType()))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MarginRankingLossForward: All input tensors should have the same type.");
        }
        return true;
    }

    bool IsAllApplicableDims() const
    {
        auto isApplicableDims = [](std::vector<std::size_t> dims) {
            size_t total_elements =
                std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
            for(size_t dim : dims)
            {
                if(total_elements == dim)
                {
                    return false;
                }
            }
            return true;
        };
        if(!isApplicableDims(input1Desc.GetLengths()) ||
           !isApplicableDims(input2Desc.GetLengths()) ||
           !isApplicableDims(targetDesc.GetLengths()) || !isApplicableDims(outputDesc.GetLengths()))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MarginRankingLossForward: Only one dim is greater than 1, there "
                         "should be at least two dims greater than 1.");
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor input1Desc;
    TensorDescriptor input2Desc;
    TensorDescriptor targetDesc;
    TensorDescriptor outputDesc;
    float margin;
    miopenMarginRakningLossReductionMode_t reduction_mode;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct ProblemDescriptionBackward : ProblemDescriptionBase
{
    ProblemDescriptionBackward(const TensorDescriptor& input1Desc_,
                               const TensorDescriptor& input2Desc_,
                               const TensorDescriptor& targetDesc_,
                               const TensorDescriptor& outGradDesc_,
                               const TensorDescriptor& in1GradDesc_,
                               const TensorDescriptor& in2GradDesc_,
                               float margin_,
                               miopenMarginRakningLossReductionMode_t reduction_mode_)
        : input1Desc(input1Desc_),
          input2Desc(input2Desc_),
          targetDesc(targetDesc_),
          outGradDesc(outGradDesc_),
          in1GradDesc(in1GradDesc_),
          in2GradDesc(in2GradDesc_),
          margin(margin_),
          reduction_mode(reduction_mode_)
    {
        IsSameLength();
        IsSameType();
        IsAllApplicableDims();
    }

    const TensorDescriptor& GetInput1Desc() const { return input1Desc; }
    const TensorDescriptor& GetInput2Desc() const { return input2Desc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetOutGradDesc() const { return outGradDesc; }
    const TensorDescriptor& GetIn1GradDesc() const { return in1GradDesc; }
    const TensorDescriptor& GetIn2GradDesc() const { return in2GradDesc; }
    const miopenMarginRakningLossReductionMode_t& GetReductionMode() const
    {
        return reduction_mode;
    }
    const float& GetMargin() const { return margin; }

    bool IsSameLength() const
    {
        auto input1Lengths  = input1Desc.GetLengths();
        auto input2Lengths  = input2Desc.GetLengths();
        auto targetLengths  = targetDesc.GetLengths();
        auto outGradLengths = outGradDesc.GetLengths();
        auto in1GradLengths = in1GradDesc.GetLengths();
        auto in2GradLengths = in2GradDesc.GetLengths();
        if((input1Lengths != input2Lengths) || (input2Lengths != targetLengths) ||
           (targetLengths != outGradLengths) || (outGradLengths != in1GradLengths) ||
           (in1GradLengths != in2GradLengths))
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "MarginRankingLossBackward: All input tensors should have the same length.");
        }
        return true;
    }

    bool IsSameType() const
    {
        if((input1Desc.GetType() != input2Desc.GetType()) ||
           (input2Desc.GetType() != targetDesc.GetType()) ||
           (targetDesc.GetType() != outGradDesc.GetType()) ||
           (outGradDesc.GetType() != in1GradDesc.GetType()) ||
           (in1GradDesc.GetType() != in2GradDesc.GetType()))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MarginRankingLossBackward: All input tensors should have the same type.");
        }
        return true;
    }

    bool IsAllApplicableDims() const
    {
        auto isApplicableDims = [](std::vector<std::size_t> dims) {
            size_t total_elements =
                std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
            for(size_t dim : dims)
            {
                if(total_elements == dim)
                {
                    return false;
                }
            }
            return true;
        };
        if(!isApplicableDims(input1Desc.GetLengths()) ||
           !isApplicableDims(input2Desc.GetLengths()) ||
           !isApplicableDims(targetDesc.GetLengths()) ||
           !isApplicableDims(outGradDesc.GetLengths()) ||
           !isApplicableDims(in1GradDesc.GetLengths()) ||
           !isApplicableDims(in2GradDesc.GetLengths()))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MarginRankingLossBackward: Only one dim is greater than 1, there "
                         "should be at least two dims greater than 1.");
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor input1Desc;
    TensorDescriptor input2Desc;
    TensorDescriptor targetDesc;
    TensorDescriptor outGradDesc;
    TensorDescriptor in1GradDesc;
    TensorDescriptor in2GradDesc;
    float margin;
    miopenMarginRakningLossReductionMode_t reduction_mode;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace marginrankingloss

} // namespace miopen
