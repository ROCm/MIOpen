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
#include <miopen/batch_norm.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>

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
    MIOPEN_LOG_FUNCTION(bn_mode,
                        xDesc,
                        x,
                        yDesc,
                        y,
                        bnScaleBiasMeanVarDesc,
                        bnScale,
                        bnBias,
                        estimatedMean,
                        estimatedVariance,
                        epsilon);

    // In case of NxCxDxHxW
    int size{0};
    miopenGetTensorDescriptorSize(xDesc, &size);
    return miopen::try_([&] {
        miopen::BatchNormForwardInference(
            miopen::deref(handle),
            bn_mode,
            alpha,
            beta,
            (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(xDesc))
                        : miopen::deref(xDesc),
            DataCast(x),
            (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(yDesc))
                        : miopen::deref(yDesc),
            DataCast(y),
            (size == 5)
                ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(bnScaleBiasMeanVarDesc))
                : miopen::deref(bnScaleBiasMeanVarDesc),
            DataCast(bnScale),
            DataCast(bnBias),
            DataCast(estimatedMean),
            DataCast(estimatedVariance),
            epsilon);
    });
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

    MIOPEN_LOG_FUNCTION(bn_mode,
                        xDesc,
                        x,
                        yDesc,
                        y,
                        bnScaleBiasMeanVarDesc,
                        bnScale,
                        bnBias,
                        expAvgFactor,
                        resultRunningMean,
                        resultRunningVariance,
                        epsilon,
                        resultSaveMean,
                        resultSaveInvVariance);

    if(miopen::IsLoggingCmd())
    {
        if(miopen::deref(xDesc).GetType() == miopenHalf)
        {
            std::cerr << MIOPEN_DRIVER_CMD("bnormfp16");
        }
        else
        {
            std::cerr << MIOPEN_DRIVER_CMD("bnorm");
        }
        std::cerr << " -n " << miopen::deref(xDesc).GetLengths()[0] << " -c "
                  << miopen::deref(xDesc).GetLengths()[1] << " -H "
                  << miopen::deref(xDesc).GetLengths()[2] << " -W "
                  << miopen::deref(xDesc).GetLengths()[3]

                  << " -m " << bn_mode;

        if((resultRunningMean != nullptr) && (resultRunningVariance != nullptr))
        {
            std::cerr << " -s " << 1;
        }
        if((resultSaveMean != nullptr) && (resultSaveInvVariance != nullptr))
        {
            std::cerr << " -r " << 1;
        }
        std::cerr << "\n";
    }

    // In case of NxCxDxHxW
    int size{0};
    miopenGetTensorDescriptorSize(xDesc, &size);
    return miopen::try_([&] {
        miopen::BatchNormForwardTraining(
            miopen::deref(handle),
            bn_mode,
            alpha,
            beta,
            (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(xDesc))
                        : miopen::deref(xDesc),
            DataCast(x),
            (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(yDesc))
                        : miopen::deref(yDesc),
            DataCast(y),
            (size == 5)
                ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(bnScaleBiasMeanVarDesc))
                : miopen::deref(bnScaleBiasMeanVarDesc),
            DataCast(bnScale),
            DataCast(bnBias),
            expAvgFactor,
            DataCast(resultRunningMean),
            DataCast(resultRunningVariance),
            epsilon,
            DataCast(resultSaveMean),
            DataCast(resultSaveInvVariance));
    });
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

    MIOPEN_LOG_FUNCTION(bn_mode,
                        xDesc,
                        x,
                        dyDesc,
                        dy,
                        dxDesc,
                        dx,
                        bnScaleBiasDiffDesc,
                        bnScale,
                        resultBnScaleDiff,
                        resultBnBiasDiff,
                        epsilon,
                        savedMean,
                        savedInvVariance);
    if(miopen::IsLoggingCmd())
    {
        if(miopen::deref(xDesc).GetType() == miopenHalf)
        {
            std::cerr << MIOPEN_DRIVER_CMD("bnormfp16");
        }
        else
        {
            std::cerr << MIOPEN_DRIVER_CMD("bnorm");
        }
        std::cerr << "\n";
    }

    // In case of NxCxDxHxW
    int size{0};
    miopenGetTensorDescriptorSize(xDesc, &size);
    return miopen::try_([&] {
        miopen::BatchNormBackward(
            miopen::deref(handle),
            bn_mode,
            alphaDataDiff,
            betaDataDiff,
            alphaParamDiff,
            betaParamDiff,
            (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(xDesc))
                        : miopen::deref(xDesc),
            DataCast(x),
            (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(dyDesc))
                        : miopen::deref(dyDesc),
            DataCast(dy),
            (size == 5) ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(dxDesc))
                        : miopen::deref(dxDesc),
            DataCast(dx),
            (size == 5)
                ? miopen::BuildReshaped4DTensorDescriptor(miopen::deref(bnScaleBiasDiffDesc))
                : miopen::deref(bnScaleBiasDiffDesc),
            DataCast(bnScale),
            DataCast(resultBnScaleDiff),
            DataCast(resultBnBiasDiff),
            epsilon,
            DataCast(savedMean),
            DataCast(savedInvVariance));
    });
}
