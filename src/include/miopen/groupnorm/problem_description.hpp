/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

namespace groupnorm {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(miopenNormMode_t mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& biasDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& meanDesc_,
                       const TensorDescriptor& rstdDesc_,
                       int32_t num_groups_,
                       float epsilon_)
        : mode(mode_),
          xDesc(xDesc_),
          weightDesc(weightDesc_),
          biasDesc(biasDesc_),
          yDesc(yDesc_),
          meanDesc(meanDesc_),
          rstdDesc(rstdDesc_),
          num_groups(num_groups_),
          epsilon(epsilon_)
    {
    }

    miopenNormMode_t GetMode() const { return mode; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetBiasDesc() const { return biasDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetMeanDesc() const { return meanDesc; }
    const TensorDescriptor& GetRstdDesc() const { return rstdDesc; }
    int32_t GetNumGroups() const { return num_groups; }
    float GetEpsilon() const { return epsilon; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "GroupNormForward: Tensor types do not match.");
        }
        if(meanDesc.GetType() != rstdDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "GroupNormForward: Tensor types do not match.");
        }
        return true;
    }

    bool IsSameLength() const
    {
        if(xDesc.GetLengths() != yDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GroupNormForward: Tensor dimension lengths do not match.");
        }
        return true;
    }

    bool IsNumGroupsValid() const
    {
        if((num_groups < 1) || (xDesc.GetLengths().size() < 2) ||
           (xDesc.GetLengths()[1] % num_groups != 0))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "GroupNormForward: The channel size of input tensor should be divisible "
                         "by num_groups.");
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && weightDesc.IsPacked() && biasDesc.IsPacked() && yDesc.IsPacked() &&
             meanDesc.IsPacked() && rstdDesc.IsPacked()))
        {
            MIOPEN_THROW(miopenStatusBadParm, "GroupNormForward: Unpacked tensors not supported.");
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    miopenNormMode_t mode;
    TensorDescriptor xDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor biasDesc;
    TensorDescriptor yDesc;
    TensorDescriptor meanDesc;
    TensorDescriptor rstdDesc;

    int32_t num_groups;
    float epsilon;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace groupnorm

} // namespace miopen
