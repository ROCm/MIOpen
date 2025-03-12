/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

namespace miopen {

struct NetworkConfig;

namespace avgpool {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_, const TensorDescriptor& outputDesc_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_)
    {
        IsValidLength();
        IsSameType();
        IsValidDims();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    auto GetNtotal() const { return outputDesc.GetElementSize(); }

    bool IsValidLength() const
    {
        auto input_dims = inputDesc.GetLengths().size();
        if(outputDesc.GetLengths()[0] != inputDesc.GetLengths()[0] ||
           outputDesc.GetLengths()[1] != inputDesc.GetLengths()[1] ||
           outputDesc.GetLengths().size() != input_dims)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "AvgPool: Input and output tensor sizes do not match.");
        }

        return true;
    }

    bool IsValidDims() const
    {
        if(inputDesc.GetLengths().size() > 5 || inputDesc.GetLengths().size() < 4)
        {
            MIOPEN_THROW(miopenStatusBadParm, "AvgPool: Only 4D and 5D tensors are supported.");
        }

        return true;
    }

    bool IsAllContiguous() const { return inputDesc.IsContiguous() && outputDesc.IsContiguous(); }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "AvgPool: Input and output tensor types do not match.");
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& inputGradDesc_)
        : outputGradDesc(outputGradDesc_), inputGradDesc(inputGradDesc_)
    {
        IsValidLength();
        IsSameType();
        IsValidDims();
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    auto GetNtotal() const { return inputGradDesc.GetElementSize(); }

    bool IsValidLength() const
    {
        auto input_dims = inputGradDesc.GetLengths().size();
        if(outputGradDesc.GetLengths()[0] != inputGradDesc.GetLengths()[0] ||
           outputGradDesc.GetLengths()[1] != inputGradDesc.GetLengths()[1] ||
           outputGradDesc.GetLengths().size() != input_dims)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "AvgPool: Input grad and output grad tensor sizes do not match.");
        }

        return true;
    }

    bool IsValidDims() const
    {
        if(inputGradDesc.GetLengths().size() > 5 || inputGradDesc.GetLengths().size() < 4)
        {
            MIOPEN_THROW(miopenStatusBadParm, "AvgPool: Only 4D and 5D tensors are supported.");
        }

        return true;
    }

    bool IsAllContiguous() const
    {
        return inputGradDesc.IsContiguous() && outputGradDesc.IsContiguous();
    }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "AvgPool: Input grad and output grad tensor types do not match.");
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor outputGradDesc;
    TensorDescriptor inputGradDesc;
};

} // namespace avgpool

} // namespace miopen
