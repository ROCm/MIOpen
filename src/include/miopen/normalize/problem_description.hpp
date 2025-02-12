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

namespace miopen {

struct NetworkConfig;

namespace normalize {

struct BackwardProblemDescription : ProblemDescriptionBase
{
    BackwardProblemDescription(const TensorDescriptor& inputDesc_,
                               const TensorDescriptor& divisorDesc_,
                               const TensorDescriptor& outputGradDesc_,
                               const TensorDescriptor& inputGradDesc_,
                               const uint32_t dim_)
        : inputDesc(inputDesc_),
          divisorDesc(divisorDesc_),
          outputGradDesc(outputGradDesc_),
          inputGradDesc(inputGradDesc_),
          dim(dim_)
    {
        if(inputDesc.GetType() != divisorDesc.GetType() ||
           inputDesc.GetType() != outputGradDesc.GetType() ||
           inputDesc.GetType() != inputGradDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "NormalizeBackward: Tensor types do not match.");
        }
        if(inputDesc.GetLengths() != outputGradDesc.GetLengths() ||
           inputDesc.GetLengths() != inputGradDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "NormalizeBackward: Input, output gradient and input gradient "
                         "tensors need to have similar shape.");
        }
        if(inputDesc.GetNumDims() != divisorDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "NormalizeBackward: Input and divisor "
                         "tensors need to have the same number of dimensions.");
        }
        for(auto i = 0; i < divisorDesc.GetNumDims(); i++)
        {
            if(i == dim)
            {
                if(divisorDesc.GetLengths()[i] != 1)
                {
                    MIOPEN_THROW(
                        miopenStatusBadParm,
                        "NormalizeBackward: The dimension 'dim' of divisor tensor should be 1.");
                }
            }
            else
            {
                if(divisorDesc.GetLengths()[i] != inputDesc.GetLengths()[i])
                {
                    MIOPEN_THROW(miopenStatusBadParm,
                                 "NormalizeBackward: The dimension != 'dim' of divisor tensor "
                                 "should equal input tensor.");
                }
            }
        }

        if(dim >= inputDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "NormalizeBackward: Dim need to be less than the number of dimensions of "
                         "input tensor.");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetDivisorDesc() const { return divisorDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    uint32_t GetDim() const { return dim; }
    bool IsLastDim() const { return (dim + 1) == inputDesc.GetNumDims(); }
    auto GetInnerSize() const { return inputDesc.GetLengths()[dim]; }
    bool IsAllContiguous() const
    {
        return inputDesc.IsContiguous() && divisorDesc.IsContiguous() &&
               outputGradDesc.IsContiguous() && inputGradDesc.IsContiguous();
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor divisorDesc;
    TensorDescriptor outputGradDesc;
    TensorDescriptor inputGradDesc;
    uint32_t dim;
};

} // namespace normalize

} // namespace miopen
