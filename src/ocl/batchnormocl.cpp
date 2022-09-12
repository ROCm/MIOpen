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

#include <miopen/check_numerics.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>
#include <miopen/util.hpp>
#include <miopen/visit_float.hpp>
/// \todo Get rid of this during implementation of #1938 (60)
#include <miopen/convolution.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batchnorm/solvers.hpp>
#include <miopen/find_solution.hpp>

#include <chrono>

namespace miopen {

void BatchNormForwardTraining(Handle& handle,
                              miopenBatchNormMode_t bn_mode,
                              const void* alpha,
                              const void* beta,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              const TensorDescriptor& bnScaleBiasMeanVarDesc,
                              ConstData_t bnScale,
                              ConstData_t bnBias,
                              double expAvgFactor,
                              Data_t resultRunningMean,
                              Data_t resultRunningVariance,
                              double epsilon,
                              Data_t resultSaveMean,
                              Data_t resultSaveInvVariance)
{

    if(x == nullptr || y == nullptr || bnScale == nullptr || bnBias == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != bnScaleBiasMeanVarDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetType() != yDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!xDesc.IsPacked())
    {
        MIOPEN_LOG_E("Only fully packed tensors supported.");
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() < 3)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0.0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    bool flag = false;
    if(miopen::CheckNumericsEnabled())
    {
        flag |= miopen::checkNumericsInput(handle, xDesc, x);
        if(bnScale != nullptr)
            flag |= miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnScale);
        if(bnBias != nullptr)
            flag |= miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnBias);
    }

    const auto resultsave    = resultSaveMean != nullptr && resultSaveInvVariance != nullptr;
    const auto resultrunning = resultRunningMean != nullptr && resultRunningVariance != nullptr;

    const auto problem = batchnorm::ProblemDescription{bn_mode,
                                                       xDesc,
                                                       yDesc,
                                                       bnScaleBiasMeanVarDesc,
                                                       expAvgFactor,
                                                       epsilon,
                                                       resultsave,
                                                       resultrunning};

    const auto algo = bn_mode == miopenBNSpatial
                          ? AlgorithmName{"miopenBatchNormForwardTrainingSpatial"}
                          : AlgorithmName{"miopenBatchNormForwardTrainingPerActivation"};

    const auto invoke_params = [&]() {
        auto tmp                  = batchnorm::InvokeParams{};
        tmp.type                  = InvokeType::Run;
        tmp.x                     = x;
        tmp.y                     = y;
        tmp.bnScale               = bnScale;
        tmp.bnBias                = bnBias;
        tmp.expAvgFactor          = expAvgFactor;
        tmp.resultRunningMean     = resultRunningMean;
        tmp.resultRunningVariance = resultRunningVariance;
        tmp.epsilon               = epsilon;
        tmp.resultSaveMean        = resultSaveMean;
        tmp.resultSaveInvVariance = resultSaveInvVariance;
        return tmp;
    }();

    const auto solvers = solver::SolverContainer<solver::batchnorm::BnFwdTrainingSpatialSingle,
                                                 solver::batchnorm::BnFwdTrainingSpatialMultiple,
                                                 solver::batchnorm::BnFwdTrainingPerActivation>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    if(miopen::CheckNumericsEnabled())
    {
        flag |= miopen::checkNumericsOutput(handle, yDesc, y);
        if(resultRunningMean != nullptr)
            flag |= miopen::checkNumericsOutput(handle, bnScaleBiasMeanVarDesc, resultRunningMean);
        if(resultRunningVariance != nullptr)
            flag |=
                miopen::checkNumericsOutput(handle, bnScaleBiasMeanVarDesc, resultRunningVariance);
        if(resultSaveMean != nullptr)
            flag |= miopen::checkNumericsOutput(handle, bnScaleBiasMeanVarDesc, resultSaveMean);
        if(resultSaveInvVariance != nullptr)
            flag |=
                miopen::checkNumericsOutput(handle, bnScaleBiasMeanVarDesc, resultSaveInvVariance);

        const char* file_name = miopen::GetStringEnv(MIOPEN_DUMP_TENSOR_PATH{});
        if(flag && static_cast<bool>(file_name))
        {
            std::string file_name_str = file_name;
            DumpTensorToFileFromDevice(handle, xDesc, x, file_name_str + "_x.bin");
            DumpTensorToFileFromDevice(handle, yDesc, y, file_name_str + "_y.bin");
            DumpTensorToFileFromDevice(
                handle, bnScaleBiasMeanVarDesc, bnScale, file_name_str + "_bnScale.bin");
            DumpTensorToFileFromDevice(
                handle, bnScaleBiasMeanVarDesc, bnBias, file_name_str + "_bnBias.bin");
            DumpTensorToFileFromDevice(handle,
                                       bnScaleBiasMeanVarDesc,
                                       resultRunningMean,
                                       file_name_str + "_resultRunningMean.bin");
            DumpTensorToFileFromDevice(handle,
                                       bnScaleBiasMeanVarDesc,
                                       resultRunningVariance,
                                       file_name_str + "_resultRunningVariance.bin");
            DumpTensorToFileFromDevice(handle,
                                       bnScaleBiasMeanVarDesc,
                                       resultSaveMean,
                                       file_name_str + "_resultSaveMean.bin");
            DumpTensorToFileFromDevice(handle,
                                       bnScaleBiasMeanVarDesc,
                                       resultSaveInvVariance,
                                       file_name_str + "_resultSaveInvVariance.bin");
        }
    }
}
//================== END FWD TRAIN ===================

//============ BEGIN FORWARD INFERENCE ===============
void BatchNormForwardInference(Handle& handle,
                               miopenBatchNormMode_t bn_mode,
                               const void* alpha,
                               const void* beta,
                               const TensorDescriptor& xDesc,
                               ConstData_t x,
                               const TensorDescriptor& yDesc,
                               Data_t y,
                               const TensorDescriptor& bnScaleBiasMeanVarDesc,
                               ConstData_t bnScale,
                               ConstData_t bnBias,
                               ConstData_t estimatedMean,
                               ConstData_t estimatedVariance,
                               double epsilon)
{
    bool flag = false;
    if(miopen::CheckNumericsEnabled())
    {
        flag |= miopen::checkNumericsInput(handle, xDesc, x);
        flag |= miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnScale);
        flag |= miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnBias);
        flag |= miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, estimatedMean);
        flag |= miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, estimatedVariance);
    }

    if(estimatedMean != nullptr && estimatedVariance != nullptr)
    {

        if(x == nullptr || y == nullptr || bnScale == nullptr || bnBias == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        if(xDesc.GetSize() != yDesc.GetSize() ||
           xDesc.GetSize() != bnScaleBiasMeanVarDesc.GetSize())
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        if(xDesc.GetType() != yDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        if(xDesc.GetSize() < 3)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
           !float_equal(*(static_cast<const float*>(beta)), 0))
        {
            MIOPEN_LOG_E("Only alpha=1 and beta=0 is supported");
            MIOPEN_THROW(miopenStatusBadParm);
        }

        const auto problem =
            batchnorm::ProblemDescription{bn_mode, xDesc, yDesc, bnScaleBiasMeanVarDesc, epsilon};

        const auto invoke_params = [&]() {
            auto tmp              = batchnorm::InfInvokeParams{};
            tmp.type              = InvokeType::Run;
            tmp.xDesc             = &xDesc;
            tmp.x                 = x;
            tmp.y                 = y;
            tmp.bnScale           = bnScale;
            tmp.bnBias            = bnBias;
            tmp.estimatedMean     = estimatedMean;
            tmp.estimatedVariance = estimatedVariance;
            tmp.epsilon           = epsilon;
            return tmp;
        }();

        const auto algo    = AlgorithmName{"miopenBatchNormalizationForwardInference"};
        const auto solvers = solver::SolverContainer<solver::batchnorm::BnFwdInference>{};

        solvers.ExecutePrimitive(handle, problem, algo, invoke_params);
    }
    else // Need to recalculated everything, let's just call training kernel in that case
    {
        MIOPEN_LOG_I2("Call to fwd train from forward inference:: ");
        BatchNormForwardTraining(handle,
                                 bn_mode,
                                 alpha,
                                 beta,
                                 xDesc,
                                 x,
                                 yDesc,
                                 y,
                                 bnScaleBiasMeanVarDesc,
                                 bnScale,
                                 bnBias,
                                 0,
                                 nullptr,
                                 nullptr,
                                 epsilon,
                                 nullptr,
                                 nullptr);
    }
    if(miopen::CheckNumericsEnabled())
    {
        flag |= miopen::checkNumericsOutput(handle, yDesc, y);
        const char* file_name = miopen::GetStringEnv(MIOPEN_DUMP_TENSOR_PATH{});
        if(flag && static_cast<bool>(file_name))
        {
            std::string file_name_str = file_name;
            DumpTensorToFileFromDevice(handle, xDesc, x, file_name_str + "_x.bin");
            DumpTensorToFileFromDevice(handle, yDesc, y, file_name_str + "_y.bin");
            DumpTensorToFileFromDevice(
                handle, bnScaleBiasMeanVarDesc, bnScale, file_name_str + "_bnScale.bin");
            DumpTensorToFileFromDevice(
                handle, bnScaleBiasMeanVarDesc, bnBias, file_name_str + "_bnBias.bin");
            DumpTensorToFileFromDevice(handle,
                                       bnScaleBiasMeanVarDesc,
                                       estimatedMean,
                                       file_name_str + "_estimatedMean.bin");
            DumpTensorToFileFromDevice(handle,
                                       bnScaleBiasMeanVarDesc,
                                       estimatedVariance,
                                       file_name_str + "_estimatedVariance.bin");
        }
    }
}
//================= END FORWARD INFERENCE ====================

//=============== BEGIN BACKWARDS PROPAGATION ================
void BatchNormBackward(Handle& handle,
                       miopenBatchNormMode_t bn_mode,
                       const void* alphaDataDiff,
                       const void* betaDataDiff,
                       const void* alphaParamDiff,
                       const void* betaParamDiff,
                       const TensorDescriptor& xDesc,
                       ConstData_t x,
                       const TensorDescriptor& dyDesc,
                       ConstData_t dy,
                       const TensorDescriptor& dxDesc,
                       Data_t dx,
                       const TensorDescriptor& bnScaleBiasDiffDesc,
                       ConstData_t bnScale,
                       Data_t resultBnScaleDiff,
                       Data_t resultBnBiasDiff,
                       double epsilon,
                       ConstData_t savedMean,
                       ConstData_t savedInvVariance)
{

#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
#endif
    bool flag = false;
    if(miopen::CheckNumericsEnabled())
    {
        flag |= miopen::checkNumericsInput(handle, xDesc, x);
        flag |= miopen::checkNumericsInput(handle, dyDesc, dy);
        flag |= miopen::checkNumericsInput(handle, bnScaleBiasDiffDesc, bnScale);

        if(savedMean != nullptr)
            flag |= miopen::checkNumericsInput(handle, bnScaleBiasDiffDesc, savedMean);
        if(savedInvVariance != nullptr)
            flag |= miopen::checkNumericsInput(handle, bnScaleBiasDiffDesc, savedInvVariance);
    }

    if(x == nullptr || dy == nullptr || bnScale == nullptr || dx == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() != dyDesc.GetSize() || xDesc.GetSize() != bnScaleBiasDiffDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dxDesc.GetType() != dyDesc.GetType() || dyDesc.GetType() != xDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() < 3)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alphaDataDiff)), 1.0) ||
       !float_equal(*(static_cast<const float*>(betaDataDiff)), 0))
    {
        MIOPEN_LOG_E("Only alphaDataDiff=1 and betaDataDiff=0 is supported");
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alphaParamDiff)), 1.0) ||
       !float_equal(*(static_cast<const float*>(betaParamDiff)), 0))
    {
        MIOPEN_LOG_E("Only alphaParamDiff=1 and betaParamDiff=0 is supported");
        MIOPEN_THROW(miopenStatusBadParm);
    }

    const auto useSaved = savedMean != nullptr && savedInvVariance != nullptr;

    const auto problem = batchnorm::ProblemDescription{
        bn_mode, xDesc, dyDesc, dxDesc, bnScaleBiasDiffDesc, epsilon, useSaved};

    const auto algo = bn_mode == miopenBNSpatial
                          ? AlgorithmName{"miopenBatchNormBackwardPropSpatial"}
                          : AlgorithmName{"miopenBatchNormBackwardPropPerActivation"};

    const auto invoke_params = [&]() {
        auto tmp              = batchnorm::BwdInvokeParams{};
        tmp.type              = InvokeType::Run;
        tmp.x                 = x;
        tmp.dy                = dy;
        tmp.dx                = dx;
        tmp.bnScale           = bnScale;
        tmp.resultBnScaleDiff = resultBnScaleDiff;
        tmp.resultBnScaleDiff = resultBnScaleDiff;
        tmp.resultBnBiasDiff  = resultBnBiasDiff;
        tmp.epsilon           = epsilon;
        tmp.savedMean         = savedMean;
        tmp.savedInvVariance  = savedInvVariance;
        return tmp;
    }();

    const auto solvers = solver::SolverContainer<solver::batchnorm::BnBwdTrainingSpatialSingle,
                                                 solver::batchnorm::BnBwdTrainingSpatialMultiple,
                                                 solver::batchnorm::BnBwdTrainingPerActivation>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    if(miopen::CheckNumericsEnabled())
    {
        flag |= miopen::checkNumericsOutput(handle, dxDesc, dx);
        flag |= miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnScaleDiff);
        flag |= miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnBiasDiff);

        const char* file_name = miopen::GetStringEnv(MIOPEN_DUMP_TENSOR_PATH{});
        if(flag && static_cast<bool>(file_name))
        {
            std::string file_name_str = file_name;
            DumpTensorToFileFromDevice(handle, xDesc, x, file_name_str + "_x.bin");
            DumpTensorToFileFromDevice(handle, dxDesc, dx, file_name_str + "_dx.bin");
            DumpTensorToFileFromDevice(handle, dyDesc, dy, file_name_str + "_dy.bin");
            DumpTensorToFileFromDevice(
                handle, bnScaleBiasDiffDesc, bnScale, file_name_str + "_bnScale.bin");
            DumpTensorToFileFromDevice(
                handle, bnScaleBiasDiffDesc, savedMean, file_name_str + "_savedMean.bin");
            DumpTensorToFileFromDevice(handle,
                                       bnScaleBiasDiffDesc,
                                       savedInvVariance,
                                       file_name_str + "_savedInvVariance.bin");
            DumpTensorToFileFromDevice(handle,
                                       bnScaleBiasDiffDesc,
                                       resultBnScaleDiff,
                                       file_name_str + "_resultBnScaleDiff.bin");
            DumpTensorToFileFromDevice(handle,
                                       bnScaleBiasDiffDesc,
                                       resultBnBiasDiff,
                                       file_name_str + "_resultBnBiasDiff.bin");
        }
    }
}
} // namespace miopen
