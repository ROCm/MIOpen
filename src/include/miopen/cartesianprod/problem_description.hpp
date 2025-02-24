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

namespace cartesianprod {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(uint64_t inputCount_,
                          const TensorDescriptor* const* inputDescs_,
                          const TensorDescriptor& outputDesc_)
        : inputDescs(inputDescs_), outputDesc(outputDesc_), inputCount(inputCount_)
    {
        IsValidNumInputs();
        IsSameType();
        IsValidDims();
        IsAllPacked();
    }

    const TensorDescriptor& GetInputDesc(uint64_t i) const
    {
        if(i >= inputCount)
        {
            MIOPEN_THROW(miopenStatusBadParm, "CartesianProdForward: Invalid tensor index.");
        }
        return deref(inputDescs[i]);
    }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    uint64_t GetInputCount() const { return inputCount; }

    bool IsValidNumInputs() const
    {
        if(inputCount < 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CartesianProdForward: At least one input tensors are required.");
        }
        return true;
    }

    bool IsSameType() const
    {
        const auto dtype = outputDesc.GetType();
        for(size_t i = 0; i < inputCount; i++)
        {
            if(GetInputDesc(i).GetType() != dtype)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CartesianProdForward: Tensor types do not match.");
            }
        }
        return true;
    }

    bool IsValidDims() const
    {
        for(size_t i = 0; i < inputCount; i++)
        {
            if(GetInputDesc(i).GetLengths().size() != 1)
            {
                MIOPEN_THROW(miopenStatusBadParm, "CartesianProdForward: Invalid input dim.");
            }
        }
        if(inputCount == 1)
        {
            if(outputDesc.GetLengths().size() != 1)
            {
                MIOPEN_THROW(miopenStatusBadParm, "CartesianProdForward: Invalid output dim.");
            }
        }
        else
        {
            if(outputDesc.GetLengths().size() != 2)
            {
                MIOPEN_THROW(miopenStatusBadParm, "CartesianProdForward: Invalid output dim.");
            }
            uint64_t outputDim0 = 1;
            for(size_t i = 0; i < inputCount; i++)
            {
                outputDim0 *= GetInputDesc(i).GetLengths()[0];
            }
            if(outputDesc.GetLengths()[0] != outputDim0 || outputDesc.GetLengths()[1] != inputCount)
            {
                MIOPEN_THROW(miopenStatusBadParm, "CartesianProdForward: Invalid output dim.");
            }
        }

        return true;
    }

    bool IsAllPacked() const
    {
        for(size_t i = 0; i < inputCount; i++)
        {
            if(!GetInputDesc(i).IsPacked())
            {
                return false;
            }
        }

        if(!outputDesc.IsPacked())
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor* const* inputDescs = nullptr;
    TensorDescriptor outputDesc{};
    uint64_t inputCount = 0;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(uint64_t inputCount_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor* const* inputGradDescs_)
        : outputGradDesc(outputGradDesc_), inputGradDescs(inputGradDescs_), inputCount(inputCount_)
    {
        IsValidNumInputs();
        IsSameType();
        IsValidDims();
        IsAllPacked();
    }

    const TensorDescriptor& GetInputGradDesc(uint64_t i) const
    {
        if(i >= inputCount)
        {
            MIOPEN_THROW(miopenStatusBadParm, "CartesianProdBackward: Invalid tensor index.");
        }
        return deref(inputGradDescs[i]);
    }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    uint64_t GetInputCount() const { return inputCount; }

    bool IsValidNumInputs() const
    {
        if(inputCount < 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "CartesianProdBackward: At least one input tensors are required.");
        }
        return true;
    }

    bool IsSameType() const
    {
        const auto dtype = outputGradDesc.GetType();
        for(size_t i = 0; i < inputCount; i++)
        {
            if(GetInputGradDesc(i).GetType() != dtype)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CartesianProdBackward: Tensor types do not match.");
            }
        }
        return true;
    }

    bool IsValidDims() const
    {
        for(size_t i = 0; i < inputCount; i++)
        {
            if(GetInputGradDesc(i).GetLengths().size() != 1)
            {
                MIOPEN_THROW(miopenStatusBadParm, "CartesianProdBackward: Invalid input grad dim.");
            }
        }
        if(inputCount == 1)
        {
            if(outputGradDesc.GetLengths().size() != 1)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CartesianProdBackward: Invalid output grad dim.");
            }
        }
        else
        {
            if(outputGradDesc.GetLengths().size() != 2)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CartesianProdBackward: Invalid output grad dim.");
            }
            uint64_t outputDim0 = 1;
            for(size_t i = 0; i < inputCount; i++)
            {
                outputDim0 *= GetInputGradDesc(i).GetLengths()[0];
            }
            if(outputGradDesc.GetLengths()[0] != outputDim0 ||
               outputGradDesc.GetLengths()[1] != inputCount)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "CartesianProdBackward: Invalid output grad dim.");
            }
        }

        return true;
    }

    bool IsAllPacked() const
    {
        for(size_t i = 0; i < inputCount; i++)
        {
            if(!GetInputGradDesc(i).IsPacked())
            {
                return false;
            }
        }

        if(!outputGradDesc.IsPacked())
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor outputGradDesc{};
    const TensorDescriptor* const* inputGradDescs = nullptr;
    uint64_t inputCount                           = 0;
};

} // namespace cartesianprod

} // namespace miopen
