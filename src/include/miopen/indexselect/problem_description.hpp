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
#include <miopen/tensor.hpp>

#include <cassert>

namespace miopen {

struct NetworkConfig;

namespace indexselect {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& indicesDesc_,
                          const TensorDescriptor& outputDesc_,
                          const size_t dim_)
        : inputDesc(inputDesc_), indicesDesc(indicesDesc_), outputDesc(outputDesc_), dim(dim_)
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(inputDesc.GetNumDims() != outputDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }

        if(indicesDesc.GetNumDims() != 1)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Indices tensor must be 1D.");
        }

        if(outputDesc.GetLengths()[dim] != indicesDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Output tensor size of dim does not match indices tensor size.");
        }

        if(indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Indices tensor must be of type miopenInt64.");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetIndices() const { return indicesDesc; }
    size_t GetDim() const { return dim; }

    bool IsAllContiguous() const
    {
        return inputDesc.IsContiguous() && outputDesc.IsContiguous() && indicesDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor indicesDesc;
    TensorDescriptor outputDesc;
    size_t dim;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& inputGradDesc_,
                          const TensorDescriptor& indicesDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const size_t dim_)
        : inputGradDesc(inputGradDesc_),
          indicesDesc(indicesDesc_),
          outputGradDesc(outputGradDesc_),
          dim(dim_)
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor types do not match.");
        }

        if(inputGradDesc.GetNumDims() != indicesDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Tensor dimension lengths do not match.");
        }

        if(indicesDesc.GetNumDims() != 1)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Indices tensor must be 1D.");
        }

        if(outputGradDesc.GetLengths()[dim] != indicesDesc.GetLengths()[0])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Output tensor size of dim does not match indices tensor size.");
        }

        if(indicesDesc.GetType() != miopenInt64)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Indices tensor must be of type miopenInt64.");
        }
    }

    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetIndices() const { return indicesDesc; }
    size_t GetDim() const { return dim; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputGradDesc;
    TensorDescriptor indicesDesc;
    TensorDescriptor outputGradDesc;
    size_t dim;
};

} // namespace indexselect

} // namespace miopen
