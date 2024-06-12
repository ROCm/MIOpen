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
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace instancenorm {

struct InstanceNormFwdProblemDescription : ProblemDescriptionBase
{
    InstanceNormFwdProblemDescription(const TensorDescriptor& inputDesc_,
                                      const TensorDescriptor& outputDesc_,
                                      const TensorDescriptor& weightDesc_,
                                      const TensorDescriptor& biasDesc_,
                                      const TensorDescriptor& meanInDesc_,
                                      const TensorDescriptor& varInDesc_,
                                      const TensorDescriptor& meanOutDesc_,
                                      const TensorDescriptor& varOutDesc_,
                                      const TensorDescriptor& meanVarDesc_,
                                      const bool useInputStats_)
        : inputDesc(inputDesc_),
          outputDesc(outputDesc_),
          weightDesc(weightDesc_),
          biasDesc(biasDesc_),
          meanInDesc(meanInDesc_),
          varInDesc(varInDesc_),
          meanOutDesc(meanOutDesc_),
          varOutDesc(varOutDesc_),
          meanVarDesc(meanVarDesc_),
          useInputStats(useInputStats_)
    {
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    const TensorDescriptor& GetWeightDesc() const { return weightDesc; }
    const TensorDescriptor& GetBiasDesc() const { return biasDesc; }
    const TensorDescriptor& GetMeanInDesc() const { return meanInDesc; }
    const TensorDescriptor& GetVarInDesc() const { return varInDesc; }
    const TensorDescriptor& GetMeanOutDesc() const { return meanOutDesc; }
    const TensorDescriptor& GetVarOutDesc() const { return varOutDesc; }
    const TensorDescriptor& GetMeanVarDesc() const { return meanVarDesc; }
    bool IsUseInputStats() const { return useInputStats; }

    NetworkConfig MakeNetworkConfig() const override;

public:
    TensorDescriptor inputDesc;
    TensorDescriptor outputDesc;
    TensorDescriptor weightDesc;
    TensorDescriptor biasDesc;
    TensorDescriptor meanInDesc;
    TensorDescriptor varInDesc;
    TensorDescriptor meanOutDesc;
    TensorDescriptor varOutDesc;
    TensorDescriptor meanVarDesc;
    bool useInputStats;
};

} // namespace instancenorm

} // namespace miopen
