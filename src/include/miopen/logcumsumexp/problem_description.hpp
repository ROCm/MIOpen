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

#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <sstream>

namespace miopen {

struct NetworkConfig;

namespace logcumsumexp {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& outputDesc_,
                              const int& dim_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_), dim(dim_)
    {
        if(IsValidDim())
            dim = (dim < 0 ? dim + inputDesc.GetNumDims() : dim);
        IsSameLength();
        IsSameType();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const int& GetDim() const { return dim; }

    bool IsValidDim() const
    {
        const int ndims = inputDesc.GetNumDims();
        if(dim < -ndims || ndims - 1 < dim)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         (std::stringstream()
                          << "LogCumSumExp: Operating dim value must be in range [" << -ndims << ","
                          << ndims - 1 << "].")
                             .str());
        }
        return true;
    }

    bool IsSameLength() const
    {
        if(inputDesc.GetLengths() != outputDesc.GetLengths())
            MIOPEN_THROW(miopenStatusBadParm,
                         "LogCumSumExp: Input and Output tensor sizes do not match.");
        return true;
    }

    bool IsSameType() const
    {
        if(inputDesc.GetType() != outputDesc.GetType())
            MIOPEN_THROW(miopenStatusBadParm,
                         "LogCumSumExp: Input and Output tensor type do not match.");
        return true;
    }

    bool IsAllPacked() const
    {
        if(!inputDesc.IsPacked() || !outputDesc.IsPacked())
            return false;
        return true;
    }

    bool IsAllDimStride1() const
    {
        if(inputDesc.GetStrides()[dim] != 1)
            return false;
        if(outputDesc.GetElementSize() > 0 && outputDesc.GetStrides()[dim] != 1)
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    int dim;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace logcumsumexp

} // namespace miopen
