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
#include <string>

namespace miopen {

struct NetworkConfig;

namespace multilabelsoftmarginloss {

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& iDesc_,
                              const TensorDescriptor& tDesc_,
                              const TensorDescriptor& wDesc_,
                              const TensorDescriptor& oDesc_,
                              const float divisor_)
        : iDesc(iDesc_), tDesc(tDesc_), wDesc(wDesc_), oDesc(oDesc_), divisor(divisor_)
    {
        if(iDesc.GetType() != tDesc.GetType() || iDesc.GetType() != oDesc.GetType() ||
           iDesc.GetType() != wDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultilabelSoftMarginLoss: Tensor types do not match.");
        }
        if(iDesc.GetSize() != 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultilabelSoftMarginLoss: Input tensor need to be 2D tensor");
        }
        if(tDesc.GetLengths() != iDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultilabelSoftMarginLoss: Tensor tensor need to be 2D tensor which is "
                         "the same shape as input tensor");
        }
        if(wDesc.GetSize() != 1 || wDesc.GetLengths()[0] != iDesc.GetLengths()[1])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MultilabelSoftMarginLoss: Weight tensor need to be 1D tensor. If input "
                         "tensor has shape (N, C) then weight tensor must have shape (C)");
        }
        // Check output tensor dimension
        if(divisor == 0)
        {
            // non-reduction case
            if(oDesc.GetSize() != 1 || oDesc.GetLengths()[0] != iDesc.GetLengths()[0])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "MultilabelSoftMarginLoss: Output tensor need to be "
                             "1D tensor. If input "
                             "tensor has shape (N, C) then output tensor must have shape (N)");
            }
        }
        else
        {
            // reduction case
            if(oDesc.GetSize() != 1 || oDesc.GetLengths()[0] != 1)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "MultilabelSoftMarginLoss: Output tensor need to be a scalar.");
            }
        }
    }

    const TensorDescriptor& GetiDesc() const { return iDesc; }
    const TensorDescriptor& GettDesc() const { return tDesc; }
    const TensorDescriptor& GetwDesc() const { return wDesc; }
    const TensorDescriptor& GetoDesc() const { return oDesc; }
    float Getdivisor() const { return divisor; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor wDesc;
    TensorDescriptor oDesc;
    float divisor;
};

} // namespace multilabelsoftmarginloss

} // namespace miopen
