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

#include "miopen/miopen.h"
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace multimarginloss {

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& iDesc_,
                              const TensorDescriptor& tDesc_,
                              const TensorDescriptor& wDesc_,
                              const TensorDescriptor& oDesc_,
                              const long p_,
                              const float margin_,
                              const miopenLossReductionMode_t reduction_)
        : iDesc(iDesc_),
          tDesc(tDesc_),
          wDesc(wDesc_),
          oDesc(oDesc_),
          p(p_),
          margin(margin_),
          reduction(reduction_)
    {
        if(iDesc.GetType() != oDesc.GetType() || iDesc.GetType() != wDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultiMarginLoss: Input, output, weight tensor types do not match.");
        }
        if(tDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultiMarginLoss: Target tensor type should be miopenInt64.");
        }
        if(iDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm, "MultiMarginLoss: Input tensor need to be 2D tensor");
        }
        if(tDesc.GetNumDims() != 1 || tDesc.GetLengths()[0] != iDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultiMarginLoss: Target tensor need to be 1D tensor. If input "
                         "tensor has shape (N, C) then target tensor must have shape (N)");
        }
        if(wDesc.GetNumDims() != 1 || wDesc.GetLengths()[0] != iDesc.GetLengths()[1])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultiMarginLoss: Weight tensor need to be 1D tensor. If input "
                         "tensor has shape (N, C) then weight tensor must have shape (C)");
        }
        // Check output tensor dimension
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            // non-reduction case
            if(oDesc.GetNumDims() != 1 || oDesc.GetLengths()[0] != iDesc.GetLengths()[0])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "MultiMarginLoss: Output tensor need to be "
                             "1D tensor. If input "
                             "tensor has shape (N, C) then output tensor must have shape (N)");
            }
        }
        else
        {
            // reduction case
            if(oDesc.GetNumDims() != 1 || oDesc.GetLengths()[0] != 1)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "MultiMarginLoss: Output tensor need to be a scalar.");
            }
        }
        // Check p value
        if(p != 1 && p != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm, "MultiMarginLoss: p need to be equal 1 or 2.");
        }
    }

    const TensorDescriptor& GetiDesc() const { return iDesc; }
    const TensorDescriptor& GettDesc() const { return tDesc; }
    const TensorDescriptor& GetwDesc() const { return wDesc; }
    const TensorDescriptor& GetoDesc() const { return oDesc; }
    long Getp() const { return p; }
    float Getmargin() const { return margin; }
    miopenLossReductionMode_t Getreduction() const { return reduction; }
    bool allContiguousTensor() const
    {
        return iDesc.IsContiguous() && tDesc.IsContiguous() && wDesc.IsContiguous() &&
               oDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor wDesc;
    TensorDescriptor oDesc;
    long p;
    float margin;
    miopenLossReductionMode_t reduction;
};

} // namespace multimarginloss

} // namespace miopen
