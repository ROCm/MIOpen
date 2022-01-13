/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/invoke_params.hpp>

namespace miopen {

namespace fusion {

struct FusionOpInvokeParamBase
{
    virtual ~FusionOpInvokeParamBase() = default;
};

struct ConvolutionOpInvokeParam : FusionOpInvokeParamBase
{
    /* Perhaps make these refs */
    TensorDescriptor workspaceDesc;
    ConstData_t workspace = nullptr;
};

struct BiasOpInvokeParam : FusionOpInvokeParamBase
{
    TensorDescriptor biasDesc;
    ConstData_t bdata = nullptr;
};

struct ActivationOpInvokeParam : FusionOpInvokeParamBase
{
    double activAlpha;
    double activBeta;
    double activGamma;
};

struct ActivationBwdOpInvokeParam : FusionOpInvokeParamBase
{
    ConstData_t y;
    ConstData_t x;
    double activAlpha;
    double activBeta;
    double activGamma;
};

struct BatchNormInferenceOpInvokeParam : FusionOpInvokeParamBase
{
    ConstData_t bnScale;
    ConstData_t bnBias;
    ConstData_t estimatedMean;
    ConstData_t estimatedVariance;
    double epsilon;
};

struct BatchNormFwdTrainingOpInvokeParam : FusionOpInvokeParamBase
{
    Data_t runningMean;
    Data_t runningVariance;
    Data_t savedMean;
    Data_t savedInvVariance;
    ConstData_t bnScale;
    ConstData_t bnBias;
    double expAvgFactor;
    double epsilon;
};

struct BatchNormBwdTrainingOpInvokeParam : FusionOpInvokeParamBase
{
    ConstData_t x;
    ConstData_t bnScale;
    ConstData_t bnBias;
    Data_t resBnScaleDiff;
    Data_t resBnBiasDiff;
    ConstData_t savedMean;
    ConstData_t savedInvVariance;
};

struct FusionInvokeParams : InvokeParams
{
    FusionInvokeParams(std::vector<std::shared_ptr<FusionOpInvokeParamBase>> op_invokers_,
                       TensorDescriptor in_desc,
                       ConstData_t in_,
                       TensorDescriptor out_desc,
                       Data_t out_,
                       bool gfx90aFp16alt_)
        : op_invokers(std::move(op_invokers_)),
          inDesc(in_desc),
          in(in_),
          outDesc(out_desc),
          out(out_),
          gfx90aFp16alt(gfx90aFp16alt_)
    {
    }

    FusionInvokeParams(InvokeType type_,
                       std::vector<std::shared_ptr<FusionOpInvokeParamBase>> op_invokers_,
                       bool gfx90aFp16alt_)
        : InvokeParams{type_}, op_invokers(std::move(op_invokers_)), gfx90aFp16alt(gfx90aFp16alt_)
    {
    }
    std::vector<std::shared_ptr<FusionOpInvokeParamBase>> op_invokers;
    TensorDescriptor inDesc;
    ConstData_t in = nullptr;
    TensorDescriptor outDesc;
    Data_t out = nullptr;
    bool gfx90aFp16alt;
};

} // namespace fusion

} // namespace miopen
