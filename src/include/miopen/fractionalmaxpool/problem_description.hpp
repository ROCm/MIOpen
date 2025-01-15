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
                          const TensorDescriptor& randomSampleDesc_,
                          const bool return_indices_,
                          const int64_t KD_,
                          const int64_t KH_,
                          const int64_t KW_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          indicesDesc(indicesDesc_),
          randomSampleDesc(randomSampleDesc_),
          return_indices(return_indices_),
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
        if(inputDesc.GetType() != outputDesc.GetType() ||
           inputDesc.GetType() != randomSampleDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm, "FractionalMaxPoolForward: Data types do not match.");
        return true;
    }

    bool IsValidType() const
    {
        if(return_indices && indicesDesc.GetType() != miopenInt32 &&
           indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: indices tensor must be int32 "
                         "or int64.");
        }
        return true;
    }

    bool IsValidDims() const
    {
        if(inputDesc.GetNumDims() != outputDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: input and output tensors must have the "
                         "same number of dimensions.");
        }

        if(return_indices && outputDesc.GetLengths() != indicesDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: output and indices tensors must have the "
                         "same dimensions.");
        }

        if(randomSampleDesc.GetNumDims() != 3)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: randomSample tensor must be 3D.");
        }

        if(inputDesc.GetNumDims() != 4 && inputDesc.GetNumDims() != 5)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: input tensor must be 4D or 5D.");
        }

        if(outputDesc.GetLengths()[2] + KD - 1 > inputDesc.GetLengths()[2] ||
           outputDesc.GetLengths()[3] + KH - 1 > inputDesc.GetLengths()[3] ||
           (inputDesc.GetNumDims() == 5 &&
            outputDesc.GetLengths()[4] + KW - 1 > inputDesc.GetLengths()[4]))
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: We must have KD + Dout - 1 <= Din, KH + Hout - "
                         "1 <= Hin, KW + Wout - "
                         "1 <= Win.");
        }
        if(randomSampleDesc.GetLengths()[2] != inputDesc.GetNumDims() - 2)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: randomSample tensor's last dimension must be "
                         "equal to (input tensor's number of dimensions - 2).");
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor indicesDesc;
    TensorDescriptor randomSampleDesc;
    bool return_indices;
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
        IsValidType();
        IsValidDims();
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolBackward: Data types do not match.");
        return true;
    }

    bool IsValidType() const
    {
        if(indicesDesc.GetType() != miopenInt32 && indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: indices tensor must be int32 "
                         "or int64.");
        }
        return true;
    }

    bool IsValidDims() const
    {
        if(inputGradDesc.GetNumDims() != outputGradDesc.GetNumDims())
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "FractionalMaxPoolForward: input grad and output grad tensors must have the "
                "same number of dimensions.");
        }

        if(outputGradDesc.GetLengths() != indicesDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: output grad and indices tensors must have the "
                         "same dimensions.");
        }
        if(inputGradDesc.GetNumDims() != 4 && inputGradDesc.GetNumDims() != 5)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "FractionalMaxPoolForward: input grad tensor must be 4D or 5D.");
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor indicesDesc;
    TensorDescriptor outputGradDesc;
    TensorDescriptor inputGradDesc;
};

} // namespace fractionalmaxpool

} // namespace miopen
