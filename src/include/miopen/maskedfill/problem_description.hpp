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

namespace maskedfill {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& inputDesc_,
                          const TensorDescriptor& outputDesc_,
                          const TensorDescriptor& maskDesc_)
        : inputDesc(inputDesc_), outputDesc(outputDesc_), maskDesc(maskDesc_)
    {
        if(inputDesc.GetLengths() != outputDesc.GetLengths() ||
           inputDesc.GetLengths() != maskDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Input tensor dimension lengths do not match.");
        }

        if(maskDesc.GetType() != miopenInt8)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MaskedFill: Mask should be a tensor of 8-bit integers.");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Input tensor type and output tensor type mismatch");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetMaskDesc() const { return maskDesc; }

    bool IsSameType() const { return inputDesc.GetType() == outputDesc.GetType(); }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor maskDesc;
};

struct BwdProblemDescription : ProblemDescriptionBase
{
    BwdProblemDescription(const TensorDescriptor& inputGradDesc_,
                          const TensorDescriptor& outputGradDesc_,
                          const TensorDescriptor& maskDesc_)
        : inputGradDesc(inputGradDesc_), outputGradDesc(outputGradDesc_), maskDesc(maskDesc_)
    {
        if(inputGradDesc.GetLengths() != outputGradDesc.GetLengths() ||
           inputGradDesc.GetLengths() != maskDesc.GetLengths())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Input tensor dimension lengths do not match.");
        }

        if(maskDesc.GetType() != miopenInt8)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "MaskedFill: Mask should be a tensor of 8-bit integers.");
        }

        if(!IsSameType())
        {
            MIOPEN_THROW(miopenStatusBadParm, "Input tensor type and output tensor type mismatch");
        }
    }

    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetMaskDesc() const { return maskDesc; }

    bool IsSameType() const { return inputGradDesc.GetType() == outputGradDesc.GetType(); }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputGradDesc;
    TensorDescriptor outputGradDesc;
    TensorDescriptor maskDesc;
};

} // namespace maskedfill

} // namespace miopen
