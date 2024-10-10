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

struct ForwardProblemDescription : ProblemDescriptionBase
{
    ForwardProblemDescription(const TensorDescriptor& inputDesc_,
                              const TensorDescriptor& outputDesc_,
                              int dim_);

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const int& GetDim() const { return dim; }

    bool IsValidDim() const;
    bool IsSameLength() const;
    bool IsSameType() const;
    bool IsSameStride() const;
    bool IsAllPacked() const;
    bool IsAllDimStride1() const;

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    int dim;
};

struct BackwardProblemDescription : ForwardProblemDescription
{
    BackwardProblemDescription(const TensorDescriptor& inputDesc_,
                               const TensorDescriptor& outputDesc_,
                               const TensorDescriptor& doutputDesc_,
                               const TensorDescriptor& dinputDesc_,
                               const int& dim_);

    const TensorDescriptor& GetDInputDesc() const { return dinputDesc; }
    const TensorDescriptor& GetDOutputDesc() const { return doutputDesc; }

    bool IsSameLength() const;
    bool IsSameType() const;
    bool IsSameStride() const;
    bool IsAllPacked() const;
    bool IsAllDimStride1() const;

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor doutputDesc;
    TensorDescriptor dinputDesc;
};

} // namespace logcumsumexp

} // namespace miopen
