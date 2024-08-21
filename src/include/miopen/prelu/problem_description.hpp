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

namespace prelu {

bool checkSameType(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);

struct BackwardProblemDescription : ProblemDescriptionBase
{

    BackwardProblemDescription(const TensorDescriptor& inputDesc_,
                               const TensorDescriptor& weightDesc_,
                               const TensorDescriptor& doutputDesc_,
                               const TensorDescriptor& dinputDesc_,
                               const TensorDescriptor& dweightDesc_)
        : inputDesc(inputDesc_),
          weightDesc(weightDesc_),
          doutputDesc(doutputDesc_),
          dinputDesc(dinputDesc_),
          dweightDesc(dweightDesc_)
    {
        IsSameType();
        IsRightLength();
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetdOuputDesc() const { return doutputDesc; }
    const TensorDescriptor& GetdInputDesc() const { return dinputDesc; }
    const TensorDescriptor& GetdWeightDesc() const { return dweightDesc; }

    bool IsSameType() const
    {
        if(!checkSameType(inputDesc, weightDesc))
            MIOPEN_THROW(miopenStatusBadParm,
                         "PReLU: Input and Weight tensor must have same type.");
        if(!checkSameType(inputDesc, dinputDesc) || !checkSameType(weightDesc, dweightDesc))
            MIOPEN_THROW(miopenStatusBadParm,
                         "PReLU: Gradient tensors (excluding Output gradient) must share a same "
                         "type with Input and Weight tensor.");
        return true;
    }

    bool IsRightLength() const
    {
        if(!checkSameLength(inputDesc, doutputDesc) || !checkSameLength(inputDesc, dinputDesc))
            MIOPEN_THROW(
                miopenStatusBadParm,
                "PReLU: Input and Output Gradient tensors sizes must match with Input tensor.");
        if(weightDesc.GetNumDims() != 1)
            MIOPEN_THROW(miopenStatusBadParm, "PReLU: Weight tensor must have 1 dimension.");
        if(weightDesc.GetElementSize() != 1 &&
           (inputDesc.GetNumDims() == 1 ||
            weightDesc.GetElementSize() != inputDesc.GetLengths()[1]))
            MIOPEN_THROW(
                miopenStatusBadParm,
                "PReLU: Weight size must be 1 or equal to the second dim of Input tensor.");
        if(!checkSameLength(weightDesc, dweightDesc))
            MIOPEN_THROW(miopenStatusBadParm,
                         "PReLU: Weight Gradient tensors sizes must match with Weight tensor.");
        return true;
    }

    bool IsSingleWeight() const
    {
        if(weightDesc.GetElementSize() > 1)
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor doutputDesc;
    TensorDescriptor dinputDesc;
    TensorDescriptor dweightDesc;
};

} // namespace prelu

} // namespace miopen
