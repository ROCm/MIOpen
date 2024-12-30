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

#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>
#include <cassert>

namespace miopen {

struct NetworkConfig;

namespace multilabelsoftmarginloss {

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& iDesc_,
                              const TensorDescriptor& tDesc_,
                              const TensorDescriptor& wDesc_,
                              const TensorDescriptor& oDesc_,
                              const miopenLossReductionMode_t reduction_)
        : iDesc(iDesc_), tDesc(tDesc_), wDesc(wDesc_), oDesc(oDesc_), reduction(reduction_)
    {
        if(iDesc.GetType() != tDesc.GetType() || iDesc.GetType() != oDesc.GetType() ||
           iDesc.GetType() != wDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultilabelSoftMarginLoss: Tensor types do not match.");
        }
        if(iDesc.GetNumDims() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultilabelSoftMarginLoss: Input tensor need to be 2D tensor");
        }
        if(tDesc.GetLengths() != iDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultilabelSoftMarginLoss: Target tensor need to be 2D tensor which is "
                         "the same shape as input tensor");
        }
        if(wDesc.GetNumDims() != 1 || wDesc.GetLengths()[0] != iDesc.GetLengths()[1])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultilabelSoftMarginLoss: Weight tensor need to be 1D tensor. If input "
                         "tensor has shape (N, C) then weight tensor must have shape (C)");
        }
        // Check output tensor dimension
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            // non-reduction case
            if(oDesc.GetNumDims() != 1 || oDesc.GetLengths()[0] != iDesc.GetLengths()[0])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "MultilabelSoftMarginLoss: When doing forward with no reduction, "
                             "output tensor need to be "
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
                             "MultilabelSoftMarginLoss: When doing forward reduction, output "
                             "tensor need to be a scalar.");
            }
        }
    }

    const TensorDescriptor& GetiDesc() const { return iDesc; }
    const TensorDescriptor& GettDesc() const { return tDesc; }
    const TensorDescriptor& GetwDesc() const { return wDesc; }
    const TensorDescriptor& GetoDesc() const { return oDesc; }
    miopenLossReductionMode_t Getreduction() const { return reduction; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor wDesc;
    TensorDescriptor oDesc;
    miopenLossReductionMode_t reduction;
};

} // namespace multilabelsoftmarginloss

} // namespace miopen
