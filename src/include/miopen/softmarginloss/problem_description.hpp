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

namespace softmarginloss {

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& iDesc_,
                              const TensorDescriptor& tDesc_,
                              const TensorDescriptor& oDesc_,
                              const miopenLossReductionMode_t reduction_)
        : iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_), reduction(reduction_)
    {
        if(iDesc.GetType() != tDesc.GetType() || iDesc.GetType() != oDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "SoftMarginLoss: Tensor types do not match.");
        }
        if(iDesc.GetLengths() != tDesc.GetLengths())
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "SoftMarginLoss: Input tensor and target tensor dimension lengths do not match.");
        }
        if((reduction == MIOPEN_LOSS_REDUCTION_NONE) && (iDesc.GetLengths() != oDesc.GetLengths()))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftMarginLoss: When doing forward with no reduction, output tensor and "
                         "input tensor dimension lengths should be equal.");
        }
        if((reduction != MIOPEN_LOSS_REDUCTION_NONE) && (oDesc.GetElementSize() != 1))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftMarginLoss: When doing forward reduction, output tensor size "
                         "need to be 1.");
        }
    }

    const TensorDescriptor& GetiDesc() const { return iDesc; }
    const TensorDescriptor& GettDesc() const { return tDesc; }
    const TensorDescriptor& GetoDesc() const { return oDesc; }
    miopenLossReductionMode_t Getreduction() const { return reduction; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor oDesc;
    miopenLossReductionMode_t reduction;
};

struct BackwardProblemDescription : ProblemDescriptionBase
{
    BackwardProblemDescription(const TensorDescriptor& iDesc_,
                               const TensorDescriptor& tDesc_,
                               const TensorDescriptor& dODesc_,
                               const TensorDescriptor& dIDesc_,
                               const miopenLossReductionMode_t reduction_)
        : iDesc(iDesc_), tDesc(tDesc_), dODesc(dODesc_), dIDesc(dIDesc_), reduction(reduction_)
    {
        if(iDesc.GetType() != tDesc.GetType() || iDesc.GetType() != dODesc.GetType() ||
           iDesc.GetType() != dIDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "SoftMarginLoss: Tensor types do not match.");
        }
        if(iDesc.GetLengths() != tDesc.GetLengths() || iDesc.GetLengths() != dODesc.GetLengths() ||
           iDesc.GetLengths() != dIDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "SoftMarginLoss: Tensor dimension lengths do not match.");
        }
    }

    const TensorDescriptor& GetiDesc() const { return iDesc; }
    const TensorDescriptor& GettDesc() const { return tDesc; }
    const TensorDescriptor& GetdODesc() const { return dODesc; }
    const TensorDescriptor& GetdIDesc() const { return dIDesc; }
    miopenLossReductionMode_t Getreduction() const { return reduction; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor dODesc;
    TensorDescriptor dIDesc;
    miopenLossReductionMode_t reduction;
};

} // namespace softmarginloss

} // namespace miopen
