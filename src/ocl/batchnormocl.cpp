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
#include <miopen/db.hpp>
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
#include <miopen/batchnorm/solvers.hpp>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/find_solution.hpp>

#include <chrono>

namespace miopen {

namespace batchnorm {
miopen::PerformanceDb GetDb(const miopen::ExecutionContext& ctx,
                            const miopen::batchnorm::ProblemDescriptionTag&)
{
    return {DbKinds::PerfDb, ctx.GetPerfDbPath("batchnorm"), ctx.GetUserPerfDbPath("batchnorm")};
}
} // namespace batchnorm

//============ BEGIN FORWARD TRAINING ===============

void BatchNormForwardTraining(Handle& handle,
                              miopenBatchNormMode_t bn_mode,
                              const void* alpha,
                              const void* beta,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              const TensorDescriptor& scaleDesc,
                              const TensorDescriptor& biasDesc,
                              const TensorDescriptor& savedMeanDesc,
                              const TensorDescriptor& savedVarianceDesc,
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
    if(xDesc.GetNumDims() != yDesc.GetNumDims() || xDesc.GetNumDims() != scaleDesc.GetNumDims() ||
       xDesc.GetNumDims() != biasDesc.GetNumDims() ||
       xDesc.GetNumDims() != savedMeanDesc.GetNumDims() ||
       xDesc.GetNumDims() != savedVarianceDesc.GetNumDims())
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
    if(xDesc.GetNumDims() < 3)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0.0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        if(bnScale != nullptr)
            miopen::checkNumericsInput(handle, scaleDesc, bnScale);
        if(bnBias != nullptr)
            miopen::checkNumericsInput(handle, biasDesc, bnBias);
    }

    const auto resultsave    = resultSaveMean != nullptr && resultSaveInvVariance != nullptr;
    const auto resultrunning = resultRunningMean != nullptr && resultRunningVariance != nullptr;

    const auto problem = batchnorm::ProblemDescription{bn_mode,
                                                       xDesc,
                                                       yDesc,
                                                       scaleDesc,
                                                       biasDesc,
                                                       savedMeanDesc,
                                                       savedVarianceDesc,
                                                       expAvgFactor,
                                                       epsilon,
                                                       resultsave,
                                                       resultrunning};

    const auto algo = bn_mode == miopenBNSpatial
                          ? AlgorithmName{"miopenBatchNormForwardTrainingSpatial"}
                          : AlgorithmName{"miopenBatchNormForwardTrainingPerActivation"};

    const auto invoke_params = [&]() {
        auto tmp                  = miopen::batchnorm::FwdTrainInvokeParams{};
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
                                                 //  solver::batchnorm::BnCKFwdTraining,
                                                 solver::batchnorm::BnFwdTrainingSpatialMultiple,
                                                 solver::batchnorm::BnFwdTrainingPerActivation>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
        if(resultRunningMean != nullptr)
            miopen::checkNumericsOutput(handle, savedMeanDesc, resultRunningMean);
        if(resultRunningVariance != nullptr)
            miopen::checkNumericsOutput(handle, savedVarianceDesc, resultRunningVariance);
        if(resultSaveMean != nullptr)
            miopen::checkNumericsOutput(handle, savedMeanDesc, resultSaveMean);
        if(resultSaveInvVariance != nullptr)
            miopen::checkNumericsOutput(handle, savedVarianceDesc, resultSaveInvVariance);
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
                               const TensorDescriptor& scaleDesc,
                               const TensorDescriptor& biasDesc,
                               const TensorDescriptor& estMeanDesc,
                               const TensorDescriptor& estVarianceDesc,
                               ConstData_t bnScale,
                               ConstData_t bnBias,
                               ConstData_t estimatedMean,
                               ConstData_t estimatedVariance,
                               double epsilon)
{

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, scaleDesc, bnScale);
        miopen::checkNumericsInput(handle, biasDesc, bnBias);
        miopen::checkNumericsInput(handle, estMeanDesc, estimatedMean);
        miopen::checkNumericsInput(handle, estVarianceDesc, estimatedVariance);
    }

    if(estimatedMean != nullptr && estimatedVariance != nullptr)
    {
        if(x == nullptr || y == nullptr || bnScale == nullptr || bnBias == nullptr)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        if(xDesc.GetNumDims() != yDesc.GetNumDims() ||
           xDesc.GetNumDims() != scaleDesc.GetNumDims() ||
           xDesc.GetNumDims() != biasDesc.GetNumDims() ||
           xDesc.GetNumDims() != estMeanDesc.GetNumDims() ||
           xDesc.GetNumDims() != estVarianceDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        if(xDesc.GetType() != yDesc.GetType())
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        if(xDesc.GetNumDims() < 3)
        {
            MIOPEN_THROW(miopenStatusBadParm);
        }
        if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
           !float_equal(*(static_cast<const float*>(beta)), 0))
        {
            MIOPEN_LOG_E("Only alpha=1 and beta=0 is supported");
            MIOPEN_THROW(miopenStatusBadParm);
        }

        const auto problem = batchnorm::ProblemDescription{
            bn_mode, xDesc, yDesc, scaleDesc, biasDesc, estMeanDesc, estVarianceDesc, epsilon};

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
        const auto solvers = solver::SolverContainer<solver::batchnorm::BnFwdInference
                                                     //  solver::batchnorm::BnCKFwdInference
                                                     >{};

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
                                 scaleDesc,
                                 biasDesc,
                                 estMeanDesc,
                                 estVarianceDesc,
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
        miopen::checkNumericsOutput(handle, yDesc, y);
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
                       const TensorDescriptor& scaleDesc,
                       const TensorDescriptor& biasDesc,
                       const TensorDescriptor& savedMeanDesc,
                       const TensorDescriptor& savedVarianceDesc,
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
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, dyDesc, dy);
        miopen::checkNumericsInput(handle, scaleDesc, bnScale);
        miopen::checkNumericsInput(handle, biasDesc, bnScale);

        if(savedMean != nullptr)
            miopen::checkNumericsInput(handle, savedMeanDesc, savedMean);
        if(savedInvVariance != nullptr)
            miopen::checkNumericsInput(handle, savedVarianceDesc, savedInvVariance);
    }

    if(x == nullptr || dy == nullptr || bnScale == nullptr || dx == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetNumDims() != dyDesc.GetNumDims() || xDesc.GetNumDims() != scaleDesc.GetNumDims() ||
       xDesc.GetNumDims() != biasDesc.GetNumDims() ||
       xDesc.GetNumDims() != savedMeanDesc.GetNumDims() ||
       xDesc.GetNumDims() != savedVarianceDesc.GetNumDims())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(dxDesc.GetType() != dyDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetNumDims() < 3)
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

    const auto problem = batchnorm::ProblemDescription{bn_mode,
                                                       xDesc,
                                                       dyDesc,
                                                       dxDesc,
                                                       scaleDesc,
                                                       biasDesc,
                                                       savedMeanDesc,
                                                       savedVarianceDesc,
                                                       epsilon,
                                                       useSaved};

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
        tmp.resultBnBiasDiff  = resultBnBiasDiff;
        tmp.epsilon           = epsilon;
        tmp.savedMean         = savedMean;
        tmp.savedInvVariance  = savedInvVariance;
        return tmp;
    }();

    const auto solvers = solver::SolverContainer<solver::batchnorm::BnBwdTrainingSpatialSingle,
                                                 //  solver::batchnorm::BnCKBwdBackward,
                                                 solver::batchnorm::BnBwdTrainingSpatialMultiple,
                                                 solver::batchnorm::BnBwdTrainingPerActivation>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
        miopen::checkNumericsOutput(handle, scaleDesc, resultBnScaleDiff);
        miopen::checkNumericsOutput(handle, biasDesc, resultBnBiasDiff);
    }
}
} // namespace miopen
