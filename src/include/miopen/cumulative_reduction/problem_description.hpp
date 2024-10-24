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
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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
#include <miopen/tensor.hpp>

#include <sstream>

namespace miopen {

struct NetworkConfig;

namespace cumulative_reduction {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& outputDesc_,
                              const TensorDescriptor& indicesDesc_,
                              const int& dim_,
                              const miopenCumOp_t& cumOp_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          indicesDesc(indicesDesc_),
          dim(dim_),
          cumOp(cumOp_)
    {
        if(IsValidDim())
            dim = (dim < 0 ? dim + inputDesc.GetNumDims() : dim);
        IsValidIndicesType();
        IsSameLength();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    const int& GetDim() const { return dim; }
    const miopenCumOp_t& GetCumOp() const { return cumOp; }

    bool IsValidDim() const
    {
        const int ndims = inputDesc.GetNumDims();
        if(dim < -ndims || ndims - 1 < dim)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         (std::stringstream()
                          << "Cumulative Reduction: Operating dim value must be in range ["
                          << -ndims << "," << ndims - 1 << "].")
                             .str());
        }
        return true;
    }

    bool IsValidIndicesType() const
    {
        if(indicesDesc.GetElementSize() > 0 && indicesDesc.GetType() != miopenInt64)
            MIOPEN_THROW(miopenStatusBadParm,
                         "Cumulative Reduction: Indices tensor type must be int32.");
        return true;
    }

    bool IsSameLength() const
    {
        if(outputDesc.GetElementSize() > 0 && !checkSameLength(inputDesc, outputDesc))
            MIOPEN_THROW(miopenStatusBadParm,
                         "Cumulative Reduction: Input and Output tensor sizes do not match.");
        if(indicesDesc.GetElementSize() > 0 && !checkSameLength(inputDesc, indicesDesc))
            MIOPEN_THROW(miopenStatusBadParm,
                         "Cumulative Reduction: Input and Indices tensor sizes do not match.");
        return true;
    }

    bool IsAllPacked() const
    {
        if(!inputDesc.IsPacked() || !outputDesc.IsPacked() || !indicesDesc.IsPacked())
            return false;
        return true;
    }

    bool IsAllDimStride1() const
    {
        if(inputDesc.GetStrides()[dim] != 1)
            return false;
        if(outputDesc.GetElementSize() > 0 && outputDesc.GetStrides()[dim] != 1)
            return false;
        if(indicesDesc.GetElementSize() > 0 && indicesDesc.GetStrides()[dim] != 1)
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor indicesDesc;
    int dim;
    miopenCumOp_t cumOp;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace cumulative_reduction

} // namespace miopen
