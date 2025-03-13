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

namespace adjust_brightness {

struct ProblemDescription : public ProblemDescriptionBase
{
    ProblemDescription(const TensorDescriptor& inputTensorDesc_,
                       const TensorDescriptor& outputTensorDesc_,
                       float brightness_factor_)
        : inputTensorDesc(inputTensorDesc_),
          outputTensorDesc(outputTensorDesc_),
          brightness_factor(brightness_factor_)
    {
        if(!IsSameType())
            MIOPEN_THROW("Input and output tensors have different types.");

        if(!IsSameSize())
            MIOPEN_THROW("Input and output tensors have different sizes.");
    }

    NetworkConfig MakeNetworkConfig() const override;

    const TensorDescriptor& GetInputTensorDesc() const { return inputTensorDesc; }
    const TensorDescriptor& GetOutputTensorDesc() const { return outputTensorDesc; }
    float GetBrightnessFactor() const { return brightness_factor; }

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

private:
    TensorDescriptor inputTensorDesc;
    TensorDescriptor outputTensorDesc;
    float brightness_factor;
};

} // namespace adjust_brightness

} // namespace image_transform

} // namespace miopen
