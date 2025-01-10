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
#include <miopen/problem_description_base.hpp>
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace fractionalmaxpool {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputDesc_,
                          const TensorDescriptor& indicesDesc_,
                          const int64_t KD_,
                          const int64_t KH_,
                          const int64_t KW_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          indicesDesc(indicesDesc_),
          KD(KD_),
          KH(KH_),
          KW(KW_)
    {
        IsSameType();
        IsValidType();
        IsValidDims();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm, "FractionalMaxPoolForward: Data types do not match.");
        return true;
    }

    bool IsValidType() const
    {
        if(indicesDesc.GetType() != miopenInt32 && indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: target tensor must be int32 "
                         "or int64.");
        }
        return true;
    }

    bool IsValidDims() const { return true; }

    bool IsAllContiguous() const
    {
        return inputDesc.IsContiguous() && outputDesc.IsContiguous() && indicesDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor indicesDesc;
    int64_t KD, KH, KW;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& indicesDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& inputGradDesc_)
        : indicesDesc(indicesDesc_), outputGradDesc(outputGradDesc_), inputGradDesc(inputGradDesc_)
    {
        IsSameType();
        IsValidDims();
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolBackward: Data types do not match.");
        return true;
    }

    bool IsValidDims() const { return true; }

    bool IsAllContiguous() const
    {
        return indicesDesc.IsContiguous() && outputGradDesc.IsContiguous() &&
               inputGradDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor indicesDesc;
    TensorDescriptor outputGradDesc;
    TensorDescriptor inputGradDesc;
};

} // namespace fractionalmaxpool

} // namespace miopen
