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
#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace nllloss {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputDesc_,
                       const TensorDescriptor& targetDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& outputDesc_,
                       int32_t ignore_index_,
                       bool is_fwd_,
                       const miopenLossReductionMode_t reduction_)
        : inputDesc(inputDesc_),
          targetDesc(targetDesc_),
          weightDesc(weightDesc_),
          outputDesc(outputDesc_),
          ignore_index(ignore_index_),
          is_fwd(is_fwd_),
          reduction(reduction_)

    {
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetTargetDesc() const { return targetDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    int32_t GetIgnoreIndex() const { return ignore_index; }
    size_t GetNtotal() const { return targetDesc.GetElementSize(); }
    size_t GetC() const { return weightDesc.GetElementSize(); }
    miopenLossReductionMode_t GetReduction() const { return reduction; }

    bool IsValidLength() const
    {
        if(reduction != MIOPEN_LOSS_REDUCTION_NONE &&
           (outputDesc.GetNumDims() != 1 || outputDesc.GetLengths()[0] != 1))
            MIOPEN_THROW(miopenStatusBadParm,
                         "NLLLoss: Output Tensor size must be (1) in Reduce mode.");

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

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor targetDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor outputDesc;

    int32_t ignore_index;
    bool is_fwd;
    miopenLossReductionMode_t reduction;
};

} // namespace nllloss

} // namespace miopen
