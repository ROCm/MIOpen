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
#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace avgpool {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& targetDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& outputDesc_,
                       int32_t ignore_index_,
                       bool is_fwd_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          weightDesc(weightDesc_),
          outputDesc(outputDesc_),
          ignore_index(ignore_index_),
          is_fwd(is_fwd_)
    {
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    int32_t GetIgnoreIndex() const { return ignore_index; }

    bool IsValidLength() const
    {
        if(targetDesc.GetLengths()[0] != inputDesc.GetLengths()[0])
            MIOPEN_THROW(miopenStatusBadParm, "NLLLoss: Tensor sizes do not match.");

        for(int32_t i = 1; i < targetDesc.GetNumDims(); ++i)
        {
            if(targetDesc.GetLengths()[i] != inputDesc.GetLengths()[i + 1])
            {
                MIOPEN_THROW(miopenStatusBadParm, "NLLLoss: Tensor sizes do not match.");
            }
        }
        if(weightDesc.GetLengths()[0] != inputDesc.GetLengths()[1])
        {
            MIOPEN_THROW(miopenStatusBadParm, "NLLLoss: Tensor sizes do not match.");
        }
        if(inputDesc.GetLengths().size() > 5)
        {
            MIOPEN_THROW(miopenStatusBadParm, "NLLLoss: Do not support Input Tensor dims > 5.");
        }
        return true;
    }

    bool IsValidStride() const
    {
        auto isRightStride = [](TensorDescriptor td) {
            auto strides = td.GetStrides();
            auto lengths = td.GetLengths();
            std::vector<std::pair<size_t, size_t>> p;
            p.reserve(td.GetNumDims());
            std::transform(strides.begin(),
                           strides.end(),
                           lengths.begin(),
                           std::back_inserter(p),
                           [](size_t a, size_t b) { return std::make_pair(a, b); });
            std::sort(p.begin(), p.end());
            for(int i = 1; i < p.size(); ++i)
            {
                if(p[i].first != p[i - 1].first * p[i - 1].second)
                    MIOPEN_THROW(miopenStatusBadParm, "NLLLoss: Tensor strides do not valid.");
            }
            return true;
        };
        return isRightStride(inputDesc) && isRightStride(targetDesc) && isRightStride(outputDesc) &&
               isRightStride(weightDesc);
    }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != weightDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "NLLLoss: Input and Weight tensors types do not match.");
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        auto isContiguous = [](TensorDescriptor td) {
            size_t s = 1;
            for(int i = td.GetNumDims() - 1; i >= 0; --i)
            {
                if(s != td.GetStrides()[i])
                {
                    return false;
                }
                s *= td.GetLengths()[i];
            }
            return true;
        };
        return isContiguous(inputDesc) && isContiguous(targetDesc) && isContiguous(weightDesc) &&
               isContiguous(outputDesc);
    }

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor outputDesc;

    int32_t ignore_index;
    bool is_fwd;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct UnreduceProblemDescription : ProblemDescription
{
    UnreduceProblemDescription(const TensorDescriptor& inputDesc_,
                               const TensorDescriptor& targetDesc_,
                               const TensorDescriptor& weightDesc_,
                               const TensorDescriptor& outputDesc_,
                               int32_t ignore_index_,
                               bool is_fwd_)
        : ProblemDescription(
              inputDesc_, targetDesc_, weightDesc_, outputDesc_, ignore_index_, is_fwd_)
    {
        IsSameType();
        IsValidLength();
        IsValidStride();
    }

    size_t GetNtotal() const { return outputDesc.GetElementSize(); }
    size_t GetC() const { return weightDesc.GetElementSize(); }

    NetworkConfig MakeNetworkConfig() const override;

private:
    NetworkConfig MakeForwardNetworkConfig() const;
};

struct ReduceProblemDescription : ProblemDescription
{
    ReduceProblemDescription(const TensorDescriptor& inputDesc_,
                             const TensorDescriptor& targetDesc_,
                             const TensorDescriptor& weightDesc_,
                             const TensorDescriptor& outputDesc_,
                             int32_t ignore_index_,
                             float divisor_,
                             bool is_fwd_)
        : ProblemDescription(
              inputDesc_, targetDesc_, weightDesc_, outputDesc_, ignore_index_, is_fwd_)
    {
        divisor = divisor_;
        IsSameType();
        IsValidLength();
        IsValidStride();
    }

    size_t GetNtotal() const { return targetDesc.GetElementSize(); }
    size_t GetC() const { return weightDesc.GetElementSize(); }

    bool IsValidLength() const
    {
        if(outputDesc.GetNumDims() != 1 || outputDesc.GetLengths()[0] != 1)
            MIOPEN_THROW(miopenStatusBadParm, "NLLLoss: Output Tensor size must be (1).");
        if(!ProblemDescription::IsValidLength())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    float divisor;
    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace avgpool

} // namespace miopen
