/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#include "miopen/common.hpp"
#include <miopen/batch_norm.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/activ.hpp>
#include <miopen/driver_arguments.hpp>

#include <array>
#include <initializer_list>

extern "C" miopenStatus_t miopenDeriveBNTensorDescriptor(miopenTensorDescriptor_t derivedBnDesc,
                                                         const miopenTensorDescriptor_t xDesc,
                                                         miopenBatchNormMode_t bn_mode)
{

    MIOPEN_LOG_FUNCTION(derivedBnDesc, xDesc, bn_mode);
    return miopen::try_([&] {
        miopen::DeriveBNTensorDescriptor(
            miopen::deref(derivedBnDesc), miopen::deref(xDesc), bn_mode);
    });
}

namespace miopen {
namespace debug {

void LogCmdBNorm(const miopenTensorDescriptor_t xDesc,
                 const miopenTensorDescriptor_t yDesc,
                 const miopenTensorDescriptor_t scaleDesc,
                 const miopenTensorDescriptor_t biasDesc,
                 const miopenTensorDescriptor_t saveMeanDesc,
                 miopenBatchNormMode_t bn_mode,
                 const void* resultRunningMean,
                 const void* resultRunningVariance,
                 const void* resultSaveMean,
                 const void* resultSaveInvVariance,
                 const BatchNormDirection_t dir,
                 const miopenActivationDescriptor_t activDesc)
{
    if(miopen::IsLoggingCmd())
    {
        const std::string& str = BnormArgsForMIOpenDriver(xDesc,
                                                          yDesc,
                                                          scaleDesc,
                                                          biasDesc,
                                                          saveMeanDesc,
                                                          bn_mode,
                                                          resultRunningMean,
                                                          resultRunningVariance,
                                                          resultSaveMean,
                                                          resultSaveInvVariance,
                                                          dir,
                                                          activDesc);
        MIOPEN_LOG_DRIVER_CMD(str);
    }
}

} // namespace debug
} // namespace miopen

extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference(miopenHandle_t handle,
                                         miopenBatchNormMode_t bn_mode,
                                         void* alpha,
                                         void* beta,
                                         const miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         const miopenTensorDescriptor_t yDesc,
                                         void* y,
                                         const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                         void* bnScale,
                                         void* bnBias,
                                         void* estimatedMean,
                                         void* estimatedVariance,
                                         double epsilon)
{
    return miopenBatchNormalizationForwardInference_V2(handle,
                                                       bn_mode,
                                                       alpha,
                                                       beta,
                                                       xDesc,
                                                       x,
                                                       yDesc,
                                                       y,
                                                       bnScaleBiasMeanVarDesc,
                                                       bnScaleBiasMeanVarDesc,
                                                       bnScaleBiasMeanVarDesc,
                                                       bnScaleBiasMeanVarDesc,
                                                       bnScale,
                                                       bnBias,
                                                       estimatedMean,
                                                       estimatedVariance,
                                                       epsilon);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining(miopenHandle_t handle,
                                        miopenBatchNormMode_t bn_mode,
                                        void* alpha,
                                        void* beta,
                                        const miopenTensorDescriptor_t xDesc,
                                        const void* x,
                                        const miopenTensorDescriptor_t yDesc,
                                        void* y,
                                        const miopenTensorDescriptor_t bnScaleBiasMeanVarDesc,
                                        void* bnScale,
                                        void* bnBias,
                                        double expAvgFactor,
                                        void* resultRunningMean,
                                        void* resultRunningVariance,
                                        double epsilon,
                                        void* resultSaveMean,
                                        void* resultSaveInvVariance)
{
    return miopenBatchNormalizationForwardTraining_V2(handle,
                                                      bn_mode,
                                                      alpha,
                                                      beta,
                                                      xDesc,
                                                      x,
                                                      yDesc,
                                                      y,
                                                      bnScaleBiasMeanVarDesc,
                                                      bnScaleBiasMeanVarDesc,
                                                      bnScaleBiasMeanVarDesc,
                                                      bnScaleBiasMeanVarDesc,
                                                      bnScale,
                                                      bnBias,
                                                      expAvgFactor,
                                                      resultRunningMean,
                                                      resultRunningVariance,
                                                      epsilon,
                                                      resultSaveMean,
                                                      resultSaveInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormalizationBackward(miopenHandle_t handle,
                                 miopenBatchNormMode_t bn_mode,
                                 const void* alphaDataDiff,
                                 const void* betaDataDiff,
                                 const void* alphaParamDiff,
                                 const void* betaParamDiff,
                                 const miopenTensorDescriptor_t xDesc,
                                 const void* x,
                                 const miopenTensorDescriptor_t dyDesc,
                                 const void* dy,
                                 const miopenTensorDescriptor_t dxDesc,
                                 void* dx,
                                 const miopenTensorDescriptor_t bnScaleBiasDiffDesc,
                                 const void* bnScale,
                                 void* resultBnScaleDiff,
                                 void* resultBnBiasDiff,
                                 double epsilon,
                                 const void* savedMean,
                                 const void* savedInvVariance)
{
    return miopenBatchNormalizationBackward_V2(handle,
                                               bn_mode,
                                               alphaDataDiff,
                                               betaDataDiff,
                                               alphaParamDiff,
                                               betaParamDiff,
                                               xDesc,
                                               x,
                                               dyDesc,
                                               dy,
                                               dxDesc,
                                               dx,
                                               bnScaleBiasDiffDesc,
                                               bnScaleBiasDiffDesc,
                                               bnScaleBiasDiffDesc,
                                               bnScaleBiasDiffDesc,
                                               bnScale,
                                               resultBnScaleDiff,
                                               resultBnBiasDiff,
                                               epsilon,
                                               savedMean,
                                               savedInvVariance);
}

extern "C" miopenStatus_t
miopenBatchNormalizationBackward_V2(miopenHandle_t handle,
                                    miopenBatchNormMode_t bn_mode,
                                    const void* alphaDataDiff,
                                    const void* betaDataDiff,
                                    const void* alphaParamDiff,
                                    const void* betaParamDiff,
                                    const miopenTensorDescriptor_t xDesc,
                                    const void* x,
                                    const miopenTensorDescriptor_t dyDesc,
                                    const void* dy,
                                    const miopenTensorDescriptor_t dxDesc,
                                    void* dx,
                                    const miopenTensorDescriptor_t scaleDesc,
                                    const miopenTensorDescriptor_t biasDesc,
                                    const miopenTensorDescriptor_t savedMeanDesc,
                                    const miopenTensorDescriptor_t savedVarianceDesc,
                                    const void* bnScale,
                                    void* resultBnScaleDiff,
                                    void* resultBnBiasDiff,
                                    double epsilon,
                                    const void* savedMean,
                                    const void* savedInvVariance)
{
    return miopenBatchNormBackwardActivation(handle,
                                             bn_mode,
                                             alphaDataDiff,
                                             betaDataDiff,
                                             alphaParamDiff,
                                             betaParamDiff,
                                             xDesc,
                                             x,
                                             dyDesc,
                                             dy,
                                             dxDesc,
                                             dx,
                                             scaleDesc,
                                             biasDesc,
                                             savedMeanDesc,
                                             savedVarianceDesc,
                                             bnScale,
                                             nullptr,
                                             resultBnScaleDiff,
                                             resultBnBiasDiff,
                                             epsilon,
                                             savedMean,
                                             savedInvVariance,
                                             nullptr);
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardInference_V2(miopenHandle_t handle,
                                            miopenBatchNormMode_t bn_mode,
                                            void* alpha,
                                            void* beta,
                                            const miopenTensorDescriptor_t xDesc,
                                            const void* x,
                                            const miopenTensorDescriptor_t yDesc,
                                            void* y,
                                            const miopenTensorDescriptor_t scaleDesc,
                                            const miopenTensorDescriptor_t biasDesc,
                                            const miopenTensorDescriptor_t estMeanDesc,
                                            const miopenTensorDescriptor_t estVarianceDesc,
                                            void* bnScale,
                                            void* bnBias,
                                            void* estimatedMean,
                                            void* estimatedVariance,
                                            double epsilon)
{
    return miopenBatchNormForwardInferenceActivation(handle,
                                                     bn_mode,
                                                     alpha,
                                                     beta,
                                                     xDesc,
                                                     x,
                                                     yDesc,
                                                     y,
                                                     scaleDesc,
                                                     biasDesc,
                                                     estMeanDesc,
                                                     estVarianceDesc,
                                                     bnScale,
                                                     bnBias,
                                                     estimatedMean,
                                                     estimatedVariance,
                                                     epsilon,
                                                     nullptr);
}

extern "C" miopenStatus_t
miopenBatchNormForwardInferenceActivation(miopenHandle_t handle,
                                          miopenBatchNormMode_t bn_mode,
                                          void* alpha,
                                          void* beta,
                                          const miopenTensorDescriptor_t xDesc,
                                          const void* x,
                                          const miopenTensorDescriptor_t yDesc,
                                          void* y,
                                          const miopenTensorDescriptor_t scaleDesc,
                                          const miopenTensorDescriptor_t biasDesc,
                                          const miopenTensorDescriptor_t estMeanDesc,
                                          const miopenTensorDescriptor_t estVarianceDesc,
                                          void* bnScale,
                                          void* bnBias,
                                          void* estimatedMean,
                                          void* estimatedVariance,
                                          double epsilon,
                                          const miopenActivationDescriptor_t activDesc)
{
    MIOPEN_LOG_FUNCTION(handle,
                        bn_mode,
                        xDesc,
                        x,
                        yDesc,
                        y,
                        scaleDesc,
                        biasDesc,
                        estMeanDesc,
                        estVarianceDesc,
                        bnScale,
                        bnBias,
                        estimatedMean,
                        estimatedVariance,
                        epsilon,
                        activDesc);

    miopen::debug::LogCmdBNorm(xDesc,
                               yDesc,
                               scaleDesc,
                               biasDesc,
                               estMeanDesc,
                               bn_mode,
                               nullptr,
                               nullptr,
                               estMeanDesc,
                               estimatedVariance,
                               miopen::debug::BatchNormDirection_t::ForwardInference,
                               activDesc);
    int size{0};
    miopenGetTensorDescriptorSize(xDesc, &size);
    // In case of NxCxDxHxW
    auto ReshapeIfNeeded = [size](const auto desc) {
        return (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(desc))
                           : miopen::deref(desc);
    };

    if(activDesc != nullptr)
    {
        miopen::ActivationDescriptor actDesc;
        actDesc = miopen::deref(activDesc);

        return miopen::try_([&] {
            miopen::BatchNormForwardInference(miopen::deref(handle),
                                              bn_mode,
                                              alpha,
                                              beta,
                                              ReshapeIfNeeded(xDesc),
                                              DataCast(x),
                                              ReshapeIfNeeded(yDesc),
                                              DataCast(y),
                                              ReshapeIfNeeded(scaleDesc),
                                              ReshapeIfNeeded(biasDesc),
                                              ReshapeIfNeeded(estMeanDesc),
                                              ReshapeIfNeeded(estVarianceDesc),
                                              DataCast(bnScale),
                                              DataCast(bnBias),
                                              DataCast(estimatedMean),
                                              DataCast(estimatedVariance),
                                              epsilon,
                                              actDesc);
        });
    }
    else
    {
        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        return miopen::try_([&] {
            miopen::BatchNormForwardInference(miopen::deref(handle),
                                              bn_mode,
                                              alpha,
                                              beta,
                                              ReshapeIfNeeded(xDesc),
                                              DataCast(x),
                                              ReshapeIfNeeded(yDesc),
                                              DataCast(y),
                                              ReshapeIfNeeded(scaleDesc),
                                              ReshapeIfNeeded(biasDesc),
                                              ReshapeIfNeeded(estMeanDesc),
                                              ReshapeIfNeeded(estVarianceDesc),
                                              DataCast(bnScale),
                                              DataCast(bnBias),
                                              DataCast(estimatedMean),
                                              DataCast(estimatedVariance),
                                              epsilon,
                                              actDesc);
        });
    }
}

extern "C" miopenStatus_t
miopenBatchNormalizationForwardTraining_V2(miopenHandle_t handle,
                                           miopenBatchNormMode_t bn_mode,
                                           void* alpha,
                                           void* beta,
                                           const miopenTensorDescriptor_t xDesc,
                                           const void* x,
                                           const miopenTensorDescriptor_t yDesc,
                                           void* y,
                                           const miopenTensorDescriptor_t scaleDesc,
                                           const miopenTensorDescriptor_t biasDesc,
                                           const miopenTensorDescriptor_t savedMeanDesc,
                                           const miopenTensorDescriptor_t savedVarianceDesc,
                                           void* bnScale,
                                           void* bnBias,
                                           double expAvgFactor,
                                           void* resultRunningMean,
                                           void* resultRunningVariance,
                                           double epsilon,
                                           void* resultSaveMean,
                                           void* resultSaveInvVariance)
{
    return miopenBatchNormForwardTrainingActivation(handle,
                                                    bn_mode,
                                                    alpha,
                                                    beta,
                                                    xDesc,
                                                    x,
                                                    yDesc,
                                                    y,
                                                    scaleDesc,
                                                    biasDesc,
                                                    savedMeanDesc,
                                                    savedVarianceDesc,
                                                    bnScale,
                                                    bnBias,
                                                    expAvgFactor,
                                                    resultRunningMean,
                                                    resultRunningVariance,
                                                    epsilon,
                                                    resultSaveMean,
                                                    resultSaveInvVariance,
                                                    nullptr);
}

extern "C" miopenStatus_t
miopenBatchNormForwardTrainingActivation(miopenHandle_t handle,
                                         miopenBatchNormMode_t bn_mode,
                                         void* alpha,
                                         void* beta,
                                         const miopenTensorDescriptor_t xDesc,
                                         const void* x,
                                         const miopenTensorDescriptor_t yDesc,
                                         void* y,
                                         const miopenTensorDescriptor_t scaleDesc,
                                         const miopenTensorDescriptor_t biasDesc,
                                         const miopenTensorDescriptor_t savedMeanDesc,
                                         const miopenTensorDescriptor_t savedVarianceDesc,
                                         void* bnScale,
                                         void* bnBias,
                                         double expAvgFactor,
                                         void* resultRunningMean,
                                         void* resultRunningVariance,
                                         double epsilon,
                                         void* resultSaveMean,
                                         void* resultSaveInvVariance,
                                         const miopenActivationDescriptor_t activDesc)
{
    MIOPEN_LOG_FUNCTION(handle,
                        bn_mode,
                        xDesc,
                        x,
                        yDesc,
                        y,
                        scaleDesc,
                        biasDesc,
                        savedMeanDesc,
                        savedVarianceDesc,
                        bnScale,
                        bnBias,
                        expAvgFactor,
                        resultRunningMean,
                        resultRunningVariance,
                        epsilon,
                        resultSaveMean,
                        resultSaveInvVariance,
                        activDesc);

    miopen::debug::LogCmdBNorm(xDesc,
                               yDesc,
                               scaleDesc,
                               biasDesc,
                               savedMeanDesc,
                               bn_mode,
                               resultRunningMean,
                               resultRunningVariance,
                               resultSaveMean,
                               resultSaveInvVariance,
                               miopen::debug::BatchNormDirection_t::ForwardTraining,
                               activDesc);

    int size{0};
    miopenGetTensorDescriptorSize(xDesc, &size);
    // In case of NxCxDxHxW
    auto ReshapeIfNeeded = [size](const auto desc) {
        return (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(desc))
                           : miopen::deref(desc);
    };

    if(activDesc != nullptr)
    {
        miopen::ActivationDescriptor actDesc;
        actDesc = miopen::deref(activDesc);

        return miopen::try_([&] {
            miopen::BatchNormForwardTraining(miopen::deref(handle),
                                             bn_mode,
                                             alpha,
                                             beta,
                                             ReshapeIfNeeded(xDesc),
                                             DataCast(x),
                                             ReshapeIfNeeded(yDesc),
                                             DataCast(y),
                                             ReshapeIfNeeded(scaleDesc),
                                             ReshapeIfNeeded(biasDesc),
                                             ReshapeIfNeeded(savedMeanDesc),
                                             ReshapeIfNeeded(savedVarianceDesc),
                                             DataCast(bnScale),
                                             DataCast(bnBias),
                                             expAvgFactor,
                                             DataCast(resultRunningMean),
                                             DataCast(resultRunningVariance),
                                             epsilon,
                                             DataCast(resultSaveMean),
                                             DataCast(resultSaveInvVariance),
                                             actDesc);
        });
    }
    else
    {
        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);

        return miopen::try_([&] {
            miopen::BatchNormForwardTraining(miopen::deref(handle),
                                             bn_mode,
                                             alpha,
                                             beta,
                                             ReshapeIfNeeded(xDesc),
                                             DataCast(x),
                                             ReshapeIfNeeded(yDesc),
                                             DataCast(y),
                                             ReshapeIfNeeded(scaleDesc),
                                             ReshapeIfNeeded(biasDesc),
                                             ReshapeIfNeeded(savedMeanDesc),
                                             ReshapeIfNeeded(savedVarianceDesc),
                                             DataCast(bnScale),
                                             DataCast(bnBias),
                                             expAvgFactor,
                                             DataCast(resultRunningMean),
                                             DataCast(resultRunningVariance),
                                             epsilon,
                                             DataCast(resultSaveMean),
                                             DataCast(resultSaveInvVariance),
                                             actDesc);
        });
    }
}

extern "C" miopenStatus_t
miopenBatchNormBackwardActivation(miopenHandle_t handle,
                                  miopenBatchNormMode_t bn_mode,
                                  const void* alphaDataDiff,
                                  const void* betaDataDiff,
                                  const void* alphaParamDiff,
                                  const void* betaParamDiff,
                                  const miopenTensorDescriptor_t xDesc,
                                  const void* x,
                                  const miopenTensorDescriptor_t dyDesc,
                                  const void* dy,
                                  const miopenTensorDescriptor_t dxDesc,
                                  void* dx,
                                  const miopenTensorDescriptor_t scaleDesc,
                                  const miopenTensorDescriptor_t biasDesc,
                                  const miopenTensorDescriptor_t savedMeanDesc,
                                  const miopenTensorDescriptor_t savedVarianceDesc,
                                  const void* bnScale,
                                  const void* bnBias,
                                  void* resultBnScaleDiff,
                                  void* resultBnBiasDiff,
                                  double epsilon,
                                  const void* savedMean,
                                  const void* savedInvVariance,
                                  const miopenActivationDescriptor_t activDesc)
{
    MIOPEN_LOG_FUNCTION(handle,
                        bn_mode,
                        xDesc,
                        x,
                        dyDesc,
                        dy,
                        dxDesc,
                        dx,
                        scaleDesc,
                        biasDesc,
                        savedMeanDesc,
                        savedVarianceDesc,
                        bnScale,
                        bnBias,
                        resultBnScaleDiff,
                        resultBnBiasDiff,
                        epsilon,
                        savedMean,
                        savedInvVariance,
                        activDesc);

    miopen::debug::LogCmdBNorm(xDesc,
                               dyDesc,
                               scaleDesc,
                               biasDesc,
                               savedMeanDesc,
                               bn_mode,
                               nullptr,
                               nullptr,
                               savedMean,
                               savedInvVariance,
                               miopen::debug::BatchNormDirection_t::Backward,
                               activDesc);

    int size{0};
    miopenGetTensorDescriptorSize(xDesc, &size);
    // In case of NxCxDxHxW
    auto ReshapeIfNeeded = [size](const auto desc) {
        return (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(desc))
                           : miopen::deref(desc);
    };

    if(activDesc != nullptr)
    {
        miopen::ActivationDescriptor actDesc;
        actDesc = miopen::deref(activDesc);

        return miopen::try_([&] {
            miopen::BatchNormBackward(miopen::deref(handle),
                                      bn_mode,
                                      alphaDataDiff,
                                      betaDataDiff,
                                      alphaParamDiff,
                                      betaParamDiff,
                                      ReshapeIfNeeded(xDesc),
                                      DataCast(x),
                                      ReshapeIfNeeded(dyDesc),
                                      DataCast(dy),
                                      ReshapeIfNeeded(dxDesc),
                                      DataCast(dx),
                                      ReshapeIfNeeded(scaleDesc),
                                      ReshapeIfNeeded(biasDesc),
                                      ReshapeIfNeeded(savedMeanDesc),
                                      ReshapeIfNeeded(savedVarianceDesc),
                                      DataCast(bnScale),
                                      DataCast(bnBias),
                                      DataCast(resultBnScaleDiff),
                                      DataCast(resultBnBiasDiff),
                                      epsilon,
                                      DataCast(savedMean),
                                      DataCast(savedInvVariance),
                                      actDesc);
        });
    }
    else
    {
        miopen::ActivationDescriptor actDesc(miopenActivationPASTHRU, 0.0f, 0.0f, 0.0f);
        return miopen::try_([&] {
            miopen::BatchNormBackward(miopen::deref(handle),
                                      bn_mode,
                                      alphaDataDiff,
                                      betaDataDiff,
                                      alphaParamDiff,
                                      betaParamDiff,
                                      ReshapeIfNeeded(xDesc),
                                      DataCast(x),
                                      ReshapeIfNeeded(dyDesc),
                                      DataCast(dy),
                                      ReshapeIfNeeded(dxDesc),
                                      DataCast(dx),
                                      ReshapeIfNeeded(scaleDesc),
                                      ReshapeIfNeeded(biasDesc),
                                      ReshapeIfNeeded(savedMeanDesc),
                                      ReshapeIfNeeded(savedVarianceDesc),
                                      DataCast(bnScale),
                                      DataCast(bnBias),
                                      DataCast(resultBnScaleDiff),
                                      DataCast(resultBnBiasDiff),
                                      epsilon,
                                      DataCast(savedMean),
                                      DataCast(savedInvVariance),
                                      actDesc);
        });
    }
}
