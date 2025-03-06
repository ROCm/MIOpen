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

#include <miopen/miopen.h>
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace kldivloss {

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& targetDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& inputGradDesc_,
                          const TensorDescriptor& targetGradDesc_,
                          bool log_target_,
                          const miopenLossReductionMode_t reduction_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          outputGradDesc(outputGradDesc_),
          inputGradDesc(inputGradDesc_),
          targetGradDesc(targetGradDesc_),
          log_target(log_target_),
          reduction(reduction_)
    {
        IsValidLength();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    size_t GetNtotal() const { return inputDesc.GetElementSize(); }
    bool GetLogTarget() const { return log_target; }
    miopenLossReductionMode_t GetReductionMode() const { return reduction; }

    bool IsValidLength() const
    {
        if(targetDesc.GetLengths() != inputDesc.GetLengths() ||
           targetDesc.GetLengths() != targetGradDesc.GetLengths() ||
           inputDesc.GetLengths() != inputGradDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "KLDivLoss: Tensor sizes do not match.");
        }

        if(inputDesc.GetNumDims() > 5)
        {
            MIOPEN_THROW(miopenStatusBadParm, "KLDivLoss: Input tensor size > 5 is not supported.");
        }

        if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        {
            if(outputGradDesc.GetNumDims() != 1 || outputGradDesc.GetLengths()[0] != 1)
                MIOPEN_THROW(miopenStatusBadParm, "KLDivLoss: Output Tensor size must be (1).");
        }
        // else
        // {
        //     if(outputGradDesc.GetNumDims() != inputDesc.GetNumDims())
        //         MIOPEN_THROW(miopenStatusBadParm,
        //                      "KLDivLoss: Output Tensor size must match input tensor size.");
        // }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor outputGradDesc;
    TensorDescriptor inputGradDesc;
    TensorDescriptor targetGradDesc;
    bool log_target;
    miopenLossReductionMode_t reduction;
};

} // namespace kldivloss

} // namespace miopen
