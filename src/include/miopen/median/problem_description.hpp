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

namespace miopen {

struct NetworkConfig;

namespace median {

struct FwdProblemDescription : public ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputDesc_,
                          const TensorDescriptor& indicesDesc_,
                          const uint64_t dim_,
                          const bool keepdim_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          indicesDesc(indicesDesc_),
          dim(dim_),
          keepdim(keepdim_)
    {
        IsValidNumDims();
        IsValidDim();
        IsRightLength();
        IsSameType();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    uint64_t GetDim() const { return dim; }
    bool GetKeepDim() const { return keepdim; }

    bool IsValidNumDims() const
    {
        if(inputDesc.GetNumDims() > 5)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "MedianForward: A tensor with dimensions greater than 5 is not supported yet.");
        }

        return true;
    }

    bool IsValidDim() const
    {

        if(dim >= inputDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MedianForward: Invalid dim " + std::to_string(dim) +
                             " for input tensor with " + std::to_string(inputDesc.GetNumDims()) +
                             " dimensions");
        }

        return true;
    }

    bool IsRightLength() const
    {
        auto input_dims   = inputDesc.GetLengths();
        auto output_dims  = outputDesc.GetLengths();
        auto indices_dims = indicesDesc.GetLengths();

        auto desired_dims = input_dims;
        if(keepdim)
        {
            desired_dims[dim] = 1;
        }
        else
        {
            desired_dims.erase(desired_dims.begin() + dim);
            if(desired_dims.size() == 0)
                desired_dims.push_back(1);
        }

        if(output_dims != desired_dims || indices_dims != desired_dims)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MedianForward: Input and output/indices tensor dimension lengths do "
                         "not match.");
        }

        return true;
    }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MedianForward: Input and output tensor data types do not match.");
        }
        return true;
    }

    bool IsValidFloat() const
    {
        return inputDesc.GetType() == miopenFloat || inputDesc.GetType() == miopenHalf ||
               inputDesc.GetType() == miopenBFloat16;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& inputDesc;
    const TensorDescriptor& outputDesc;
    const TensorDescriptor& indicesDesc;
    uint64_t dim;
    bool keepdim;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& indicesDesc_,
                          const TensorDescriptor& inputGradDesc_,
                          const uint64_t dim_,
                          const bool keepdim_)
        : outputGradDesc(outputGradDesc_),
          indicesDesc(indicesDesc_),
          inputGradDesc(inputGradDesc_),
          dim(dim_),
          keepdim(keepdim_)
    {
        IsValidNumDims();
        IsValidDim();
        IsRightLength();
        IsSameType();
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    uint64_t GetDim() const { return dim; }
    bool GetKeepDim() const { return keepdim; }

    bool IsValidNumDims() const
    {
        if(inputGradDesc.GetNumDims() > 5)
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "MedianForward: A tensor with dimensions greater than 5 is not supported yet.");
        }

        return true;
    }

    bool IsValidDim() const
    {

        if(dim >= inputGradDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MedianForward: Invalid dim " + std::to_string(dim) +
                             " for input tensor with " +
                             std::to_string(inputGradDesc.GetNumDims()) + " dimensions");
        }

        return true;
    }

    bool IsRightLength() const
    {
        auto input_grad_dims  = inputGradDesc.GetLengths();
        auto output_grad_dims = outputGradDesc.GetLengths();
        auto indices_dims     = indicesDesc.GetLengths();

        auto desired_dims = input_grad_dims;
        if(keepdim)
        {
            desired_dims[dim] = 1;
        }
        else
        {
            desired_dims.erase(desired_dims.begin() + dim);
            if(desired_dims.size() == 0)
                desired_dims.push_back(1);
        }

        if(output_grad_dims != desired_dims || indices_dims != desired_dims)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MedianForward: Input and output/indices tensor dimension lengths do "
                         "not match.");
        }

        return true;
    }

    bool IsSameType() const
    {
        if(inputGradDesc.GetType() != outputGradDesc.GetType())
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "MedianForward: Input, input grad, output grad tensor data types do not match.");
        }
        return true;
    }

    bool IsValidFloat() const
    {
        return inputGradDesc.GetType() == miopenFloat || inputGradDesc.GetType() == miopenHalf ||
               inputGradDesc.GetType() == miopenBFloat16;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor& outputGradDesc;
    const TensorDescriptor& indicesDesc;
    const TensorDescriptor& inputGradDesc;
    uint64_t dim;
    bool keepdim;
};

} // namespace median

} // namespace miopen
