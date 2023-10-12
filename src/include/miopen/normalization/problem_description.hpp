/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <miopen/activ.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace normalization {

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(miopenLayerNormMode_t mode_,
                       const TensorDescriptor& xDesc_,
                       const TensorDescriptor& weightDesc_,
                       const TensorDescriptor& biasDesc_,
                       const TensorDescriptor& yDesc_,
                       const TensorDescriptor& meanDesc_,
                       const TensorDescriptor& rstdDesc_,
                       double epsilon_,
                       int32_t normalized_dim_)
        : mode(mode_),
          xDesc(xDesc_),
          weightDesc(weightDesc_),
          biasDesc(biasDesc_),
          yDesc(yDesc_),
          meanDesc(meanDesc_),
          rstdDesc(rstdDesc_),
          epsilon_(epsilon_),
          resultrunning(normalized_dim_) {}

    miopenLayerNormMode_t GetMode() const { return mode; }
    const TensorDescriptor& GetXDesc() const { return xDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetBiasDesc() const { return biasDesc; }
    const TensorDescriptor& GetYDesc() const{ return yDesc; }
    const TensorDescriptor& GetMeanDesc() const { return meanDesc; }
    const TensorDescriptor& GetRstdDesc() const { return rstdDesc; }

    NetworkConfig MakeNetworkConfig() const;

private:
    miopenLayerNormMode_t mode;
    TensorDescriptor xDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor biasDesc;
    TensorDescriptor yDesc;
    TensorDescriptor meanDesc;
    TensorDescriptor rstdDesc;

    float epsilon;
    int32_t normalized_dim;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace normalization

} // namespace miopen
