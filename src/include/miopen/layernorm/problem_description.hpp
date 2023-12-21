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

namespace layernorm {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(miopenNormMode_t mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& biasDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& meanDesc_,
                       const TensorDescriptor& rstdDesc_,
                       float epsilon_,
                       int32_t normalized_dim_)
        : mode(mode_),
          xDesc(xDesc_),
          weightDesc(weightDesc_),
          biasDesc(biasDesc_),
          yDesc(yDesc_),
          meanDesc(meanDesc_),
          rstdDesc(rstdDesc_),
          epsilon(epsilon_),
          normalized_dim(normalized_dim_)
    {
    }

    miopenNormMode_t GetMode() const { return mode; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetBiasDesc() const { return biasDesc; }
    const TensorDescriptor& GetYDesc() const { return yDesc; }
    const TensorDescriptor& GetMeanDesc() const { return meanDesc; }
    const TensorDescriptor& GetRstdDesc() const { return rstdDesc; }
    float GetEpsilon() const { return epsilon; }
    int32_t GetNormalizedDim() const { return normalized_dim; }

    bool IsSameType() const
    {
        if(xDesc.GetType() != yDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "LayerNormForward: Tensor types do not match.");
        }
        return true;
    }

    bool IsSameLength() const
    {
        if(xDesc.GetLengths() != yDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "LayerNormForward: Tensor dimension lengths do not match.");
        }
        return true;
    }

    bool IsRightNormDim() const
    {
        if((normalized_dim < 0) || (normalized_dim > xDesc.GetLengths().size()))
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "LayerNormForward: normalized dim is greater than 0 and less than or equal "
                "Tensor dimension length.");
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(xDesc.IsPacked() && weightDesc.IsPacked() && biasDesc.IsPacked() && yDesc.IsPacked() &&
             meanDesc.IsPacked() && rstdDesc.IsPacked()))
        {
            MIOPEN_THROW(miopenStatusBadParm, "LayerNormForward: Unpacked tensors not supported.");
        }
        return true;
    }

    bool IsLargeSize() const
    {
        auto dims = xDesc.GetLengths();

        size_t outer_size = 1;
        for(size_t i = 0; i < normalized_dim; i++)
        {
            outer_size *= dims[i];
        }

        return (outer_size > 32);
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

    float epsilon;
    int32_t normalized_dim;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace layernorm

} // namespace miopen
