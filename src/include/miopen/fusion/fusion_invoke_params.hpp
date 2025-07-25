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

#include <miopen/fusion/fusion_op_args.hpp>
#include <miopen/invoke_params.hpp>

namespace miopen {

namespace fusion {

struct FusionOpInvokeParamBase
{
    virtual ~FusionOpInvokeParamBase() = default;
};

struct ConvolutionOpInvokeParam : FusionOpInvokeParamBase
{
    ConvolutionOpInvokeParam(ConstData_t w) : weights(w) {}
    ConvolutionOpInvokeParam(float _alpha, float _beta, ConstData_t w)
        : alpha(_alpha), beta(_beta), weights(w)
    {
    }
    float alpha         = 1.0f; // scales new result of convolution
    float beta          = 0.0f; // scales old val of convolution output tensor
    ConstData_t weights = nullptr;
};

struct BiasOpInvokeParam : FusionOpInvokeParamBase
{
    BiasOpInvokeParam(ConstData_t b) : bdata(b) {}
    ConstData_t bdata = nullptr;
};

struct TensorScaleAddOpInvokeParam : public FusionOpInvokeParamBase
{
    TensorScaleAddOpInvokeParam(float a, ConstData_t tp) : alpha(a), tensor_ptr(tp) {}
    float alpha            = 1.0f;
    ConstData_t tensor_ptr = nullptr;
};

struct ActivationOpInvokeParam : FusionOpInvokeParamBase
{
    ActivationOpInvokeParam(double alpha, double beta, double gamma, miopenActivationMode_t mode)
        : activAlpha(alpha), activBeta(beta), activGamma(gamma), activMode(mode)
    {
    }
    double activAlpha;
    double activBeta;
    double activGamma;
    miopenActivationMode_t activMode = miopenActivationRELU;
};

struct ActivationBwdOpInvokeParam : FusionOpInvokeParamBase
{
    ActivationBwdOpInvokeParam(
        ConstData_t _y, ConstData_t _x, double alpha, double beta, double gamma)
        : y(_y), x(_x), activAlpha(alpha), activBeta(beta), activGamma(gamma)
    {
    }
    ConstData_t y;
    ConstData_t x;
    double activAlpha;
    double activBeta;
    double activGamma;
};

struct BatchNormInferenceOpInvokeParam : FusionOpInvokeParamBase
{
    BatchNormInferenceOpInvokeParam(
        ConstData_t scale, ConstData_t bias, ConstData_t estMean, ConstData_t estVar, double eps)
        : bnScale(scale),
          bnBias(bias),
          estimatedMean(estMean),
          estimatedVariance(estVar),
          epsilon(eps)
    {
    }
    ConstData_t bnScale;
    ConstData_t bnBias;
    ConstData_t estimatedMean;
    ConstData_t estimatedVariance;
    double epsilon;
};

struct BatchNormFwdTrainingOpInvokeParam : FusionOpInvokeParamBase
{
    BatchNormFwdTrainingOpInvokeParam(Data_t rMean,
                                      Data_t rVar,
                                      Data_t sMean,
                                      Data_t sInvVar,
                                      ConstData_t scale,
                                      ConstData_t bias,
                                      double eaFact,
                                      double eps)
        : runningMean(rMean),
          runningVariance(rVar),
          savedMean(sMean),
          savedInvVariance(sInvVar),
          bnScale(scale),
          bnBias(bias),
          expAvgFactor(eaFact),
          epsilon(eps)
    {
    }
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
    BatchNormBwdTrainingOpInvokeParam(ConstData_t _x,
                                      ConstData_t scale,
                                      ConstData_t bias,
                                      Data_t scaleDiff,
                                      Data_t biasDiff,
                                      ConstData_t sMean,
                                      ConstData_t sInvVar)
        : x(_x),
          bnScale(scale),
          bnBias(bias),
          resBnScaleDiff(scaleDiff),
          resBnBiasDiff(biasDiff),
          savedMean(sMean),
          savedInvVariance(sInvVar)
    {
    }
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
    FusionInvokeParams(const miopen::OperatorArgs& op_args_,
                       TensorDescriptor in_desc,
                       ConstData_t in_,
                       TensorDescriptor out_desc,
                       Data_t out_,
                       bool gfx90aFp16alt_)
        : op_args(op_args_),
          inDesc(in_desc),
          in(in_),
          outDesc(out_desc),
          out(out_),
          gfx90aFp16alt(gfx90aFp16alt_)
    {
    }

    FusionInvokeParams(const miopen::OperatorArgs& op_args_,
                       TensorDescriptor in_desc,
                       ConstData_t in_,
                       TensorDescriptor out_desc,
                       Data_t out_,
                       bool gfx90aFp16alt_,
                       Data_t workSpace_,
                       size_t workSpaceSize_)
        : op_args(op_args_),
          inDesc(in_desc),
          in(in_),
          outDesc(out_desc),
          out(out_),
          gfx90aFp16alt(gfx90aFp16alt_),
          workSpace(workSpace_),
          workSpaceSize(workSpaceSize_)
    {
    }

    FusionInvokeParams(InvokeType type_, const OperatorArgs& op_args_, bool gfx90aFp16alt_)
        : InvokeParams{type_}, op_args(op_args_), gfx90aFp16alt(gfx90aFp16alt_)
    {
    }
    const miopen::OperatorArgs& op_args;
    TensorDescriptor inDesc;
    ConstData_t in = nullptr;
    TensorDescriptor outDesc;
    Data_t out = nullptr;
    bool gfx90aFp16alt;
    Data_t workSpace          = nullptr;
    std::size_t workSpaceSize = 0;

    Data_t GetWorkspace() const { return workSpace; }

    std::size_t GetWorkspaceSize() const { return workSpaceSize; }
};

} // namespace fusion

} // namespace miopen
