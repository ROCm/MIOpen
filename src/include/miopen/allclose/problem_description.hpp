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

namespace allclose {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& input1Desc_,
                       const TensorDescriptor& input2Desc_,
                       const TensorDescriptor& outputDesc_)
        : input1Desc(input1Desc_), input2Desc(input2Desc_), outputDesc(outputDesc_)
    {
        IsSameType();
        IsValidType();
        IsValidDims();
    }

    const TensorDescriptor& GetInput1Desc() const { return input1Desc; }
    const TensorDescriptor& GetInput2Desc() const { return input2Desc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    bool IsSameType() const
    {
        if(input1Desc.GetType() != input2Desc.GetType())
            MIOPEN_THROW(miopenStatusBadParm, "AllCloseForward: Data types do not match.");
        return true;
    }

    bool IsValidType() const
    {
        if(outputDesc.GetType() != miopenInt32)
            MIOPEN_THROW(miopenStatusBadParm, "AllCloseForward: Output type must be miopenInt32.");
        return true;
    }

    bool IsValidDims() const
    {
        if(input1Desc.GetLengths() != input2Desc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "AllCloseForward: Tensor sizes do not match.");
        }
        return true;
    }

    bool IsAllContiguous() const { return input1Desc.IsContiguous() && input2Desc.IsContiguous(); }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor input1Desc;
    TensorDescriptor input2Desc;
    TensorDescriptor outputDesc;
};

} // namespace allclose

} // namespace miopen
