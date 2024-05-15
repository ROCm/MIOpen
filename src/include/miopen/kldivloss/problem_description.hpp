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

namespace kldivloss {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& targetDesc_,
                       const TensorDescriptor& outputDesc_,
                       bool log_target_,
                       bool is_fwd_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          outputDesc(outputDesc_),
          log_target(log_target_),
          is_fwd(is_fwd_)
    {
        IsValidStride();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    size_t GetNtotal() const { return inputDesc.GetElementSize(); }
    bool GetLogTarget() const { return log_target; }

    bool IsValidLength() const
    {
        for(int32_t i = 0; i < targetDesc.GetSize(); ++i)
        {
            if(targetDesc.GetLengths()[i] != inputDesc.GetLengths()[i])
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "KLDivLoss: Tensor sizes do not match.");
#else
                return false;
#endif
            }
        }
        if(inputDesc.GetSize() > 5)
        {
            #if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "KLDivLoss: Input tensor size > 5 is not supported.");
            #else
                            return false;
            #endif
                    }
        return true;
    }

    bool IsValidStride() const
    {
        auto isRightStride = [](TensorDescriptor td) {
            auto strides = td.GetStrides();
            auto lengths = td.GetLengths();
            std::vector<std::pair<size_t, size_t>> p;
            p.reserve(td.GetSize());
            std::transform(strides.begin(),
                           strides.end(),
                           lengths.begin(),
                           std::back_inserter(p),
                           [](size_t a, size_t b) { return std::make_pair(a, b); });
            std::sort(p.begin(), p.end());
            for(int i = 1; i < p.size(); ++i)
            {
                if(p[i].first != p[i - 1].first * p[i - 1].second)
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                    MIOPEN_THROW(miopenStatusBadParm, "KLDivLoss: Tensor strides do not valid.");
#else
                    return false;
#endif
            }
            return true;
        };
        return isRightStride(inputDesc) && isRightStride(targetDesc) && isRightStride(outputDesc);
    }
    
protected:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor outputDesc;

    bool log_target;
    bool is_fwd;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct UnreducedProblemDescription : ProblemDescription
{
    UnreducedProblemDescription(const TensorDescriptor& inputDesc_,
                               const TensorDescriptor& targetDesc_,
                               const TensorDescriptor& outputDesc_,
                               bool log_target_,
                               bool is_fwd_)
        : ProblemDescription(
              inputDesc_, targetDesc_, outputDesc_, log_target_, is_fwd_)
    {
        IsValidLength();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    NetworkConfig MakeForwardNetworkConfig() const;
};

struct ReducedProblemDescription : ProblemDescription
{
    ReducedProblemDescription(const TensorDescriptor& inputDesc_,
                             const TensorDescriptor& targetDesc_,
                             const TensorDescriptor& outputDesc_,
                             bool log_target_,
                             bool is_fwd_)
        : ProblemDescription(
              inputDesc_, targetDesc_, outputDesc_, log_target_, is_fwd_)
    {
        IsValidLength();
    }

    bool IsValidLength() const
    {
        if(!ProblemDescription::IsValidLength())
            return false;
        if(outputDesc.GetSize() != 1 || outputDesc.GetLengths()[0] != 1)
            MIOPEN_THROW(miopenStatusBadParm, "KLDivLoss: Output Tensor size must be (1).");
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace kldivloss

} // namespace miopen
