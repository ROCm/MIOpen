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

#include <miopen/names.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace image_transform {

namespace normalize {

struct ProblemDescription : public ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputTensorDesc_,
                       const TensorDescriptor& meanTensorDesc_,
                       const TensorDescriptor& stddevTensorDesc_,
                       const TensorDescriptor& outputTensorDesc_)
        : inputTensorDesc(inputTensorDesc_),
          meanTensorDesc(meanTensorDesc_),
          stddevTensorDesc(stddevTensorDesc_),
          outputTensorDesc(outputTensorDesc_)
    {
        if(!IsMeanAndStdDevContiguous())
            MIOPEN_THROW("Mean and stddev tensors are not contiguous.");

        if(!IsSameType())
            MIOPEN_THROW("Input and output tensors have different types.");

        if(!IsSameSize())
            MIOPEN_THROW("Input and output tensors have different sizes.");

        if(!IsInputSizesValid())
        {
            MIOPEN_THROW("Input tensor must have 4 dimensions.");
        }
    }

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetInputTensorDesc() const { return inputTensorDesc; }
    const TensorDescriptor& GetMeanTensorDesc() const { return meanTensorDesc; }
    const TensorDescriptor& GetStddevTensorDesc() const { return stddevTensorDesc; }
    const TensorDescriptor& GetOutputTensorDesc() const { return outputTensorDesc; }

    bool IsSameType() const { return inputTensorDesc.GetType() == outputTensorDesc.GetType(); }

    bool IsSameSize() const
    {
        for(int i = 0; i < inputTensorDesc.GetLengths().size(); i++)
        {
            if(inputTensorDesc.GetLengths()[i] != outputTensorDesc.GetLengths()[i])
            {
                return false;
            }
        }

        return true;
    }
    bool IsInputSizesValid() const
    {
        if(inputTensorDesc.GetLengths().size() == 4)
            return true;

        return false;
    }

    bool IsMeanAndStdDevContiguous() const
    {
        // We cannot handle a case where they are not unfortunately
        return meanTensorDesc.IsContiguous() && stddevTensorDesc.IsContiguous();
    }

private:
    TensorDescriptor inputTensorDesc;
    TensorDescriptor meanTensorDesc;
    TensorDescriptor stddevTensorDesc;
    TensorDescriptor outputTensorDesc;
};

} // namespace normalize

} // namespace image_transform

} // namespace miopen
