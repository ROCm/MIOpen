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

#include "miopen/errors.hpp"
#include "miopen/miopen.h"
#include <cstddef>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace kthvalue {

struct KthvalueFwdProblemDescription : ProblemDescriptionBase
{
    KthvalueFwdProblemDescription(const TensorDescriptor& inputDesc_,
                                  const TensorDescriptor& outputDesc_,
                                  const TensorDescriptor& indicesDesc_,
                                  int32_t dim_,
                                  size_t k_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          indicesDesc(indicesDesc_),
          dim(dim_),
          k(k_)
    {
        if(k > inputDesc.GetLengths()[dim])
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "Kthvalue: k must be less than the size of the dimension");
        }
        int num_dim = inputDesc.GetSize();
        if(dim < -num_dim || dim >= num_dim)
        {
            MIOPEN_THROW(miopenStatusBadParm, "Kthvalue: dim doesn't not exist");
        }
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetIndicesDesc() const { return indicesDesc; }
    int32_t GetDim() const { return dim; }
    size_t GetK() const { return k; }
    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor indicesDesc;
    int32_t dim;
    size_t k;
};

} // namespace kthvalue

} // namespace miopen
