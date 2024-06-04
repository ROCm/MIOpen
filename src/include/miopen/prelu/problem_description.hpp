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
bool checkSameStride(const TensorDescriptor& x, const TensorDescriptor& y);

struct BackwardProblemDescription : ProblemDescriptionBase
{

    BackwardProblemDescription(const TensorDescriptor& inputDesc_,
                               const TensorDescriptor& weightDesc_,
                               const TensorDescriptor& outputDesc_,
                               const TensorDescriptor& doutputDesc_,
                               const TensorDescriptor& dinputDesc_,
                               const TensorDescriptor& dweightDesc_)
        : inputDesc(inputDesc_),
          weightDesc(weightDesc_),
          outputDesc(outputDesc_),
          doutputDesc(doutputDesc_),
          dinputDesc(dinputDesc_),
          dweightDesc(dweightDesc_)
    {
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetdOuputDesc() const { return doutputDesc; }
    const TensorDescriptor& GetdInputDesc() const { return dinputDesc; }
    const TensorDescriptor& GetdWeightDesc() const { return dweightDesc; }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor inputDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor doutputDesc;
    TensorDescriptor dinputDesc;
    TensorDescriptor dweightDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace prelu

} // namespace miopen
