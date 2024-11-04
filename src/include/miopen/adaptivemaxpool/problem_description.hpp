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

namespace miopen {

struct NetworkConfig;

namespace adaptivemaxpool {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputDesc_,
                          const TensorDescriptor& indicesDesc_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_), indicesDesc(indicesDesc_)
    {
        IsValidLength();
        IsValidDims();
        IsSameType();
        IsValidType();
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
                         "AdaptiveMaxPool: Input and output tensor sizes do not match.");
        }

        for(auto i = 2; i < input_dims; i++)
        {
            if(outputDesc.GetLengths()[i] > inputDesc.GetLengths()[i])
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "AdaptiveMaxPool: Input tensor sizes are too small compare to output "
                             "tensor sizes.");
            }
        }

        if(indicesDesc.GetElementSize() != 1)
        {
            if(outputDesc.GetLengths() != indicesDesc.GetLengths())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "AdaptiveMaxPool: Indices and output tensor sizes do not match.");
            }
        }

        return true;
    }

    bool IsValidDims() const
    {
        if(inputDesc.GetLengths().size() > 5 || inputDesc.GetLengths().size() < 3)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "AdaptiveMaxPool: Only 3D, 4D and 5D tensors are supported.");
        }

        return true;
    }

    bool IsAllContiguous() const
    {
        return inputDesc.IsContiguous() && outputDesc.IsContiguous() && indicesDesc.IsContiguous();
    }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "AdaptiveMaxPool: Input and output tensor types do not match.");
        }

        return true;
    }

    bool IsValidType() const
    {
        if(indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(miopenStatusBadParm, "AdaptiveMaxPool: Indices tensor should be int64.");
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor indicesDesc;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& indicesDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& inputGradDesc_)
        : indicesDesc(indicesDesc_), outputGradDesc(outputGradDesc_), inputGradDesc(inputGradDesc_)
    {
        IsValidLength();
        IsValidDims();
        IsSameType();
        IsValidType();
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
                         "AdaptiveMaxPool: Input grad and output grad tensor sizes do not match.");
        }

        for(auto i = 2; i < input_dims; i++)
        {
            if(outputGradDesc.GetLengths()[i] > inputGradDesc.GetLengths()[i])
            {
                MIOPEN_THROW(
                    miopenStatusBadParm,
                    "AdaptiveMaxPool: Input grad tensor sizes are too small compare to output grad "
                    "tensor sizes.");
            }
        }

        if(indicesDesc.GetElementSize() != 1)
        {
            if(outputGradDesc.GetLengths() != indicesDesc.GetLengths())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "AdaptiveMaxPool: Indices and output grad tensor sizes do not match.");
            }
        }

        return true;
    }

    bool IsValidDims() const
    {
        if(inputGradDesc.GetLengths().size() > 5 || inputGradDesc.GetLengths().size() < 3)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "AdaptiveMaxPool: Only 3D, 4D and 5D tensors are supported.");
        }

        return true;
    }

    bool IsAllContiguous() const
    {
        return inputGradDesc.IsContiguous() && outputGradDesc.IsContiguous() &&
               indicesDesc.IsContiguous();
    }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "AdaptiveMaxPool: Input grad and output grad tensor types do not match.");
        }

        return true;
    }

    bool IsValidType() const
    {
        if(indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(miopenStatusBadParm, "AdaptiveMaxPool: Indices tensor should be int64.");
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor indicesDesc;
    TensorDescriptor outputGradDesc;
    TensorDescriptor inputGradDesc;
};

} // namespace adaptivemaxpool

} // namespace miopen
