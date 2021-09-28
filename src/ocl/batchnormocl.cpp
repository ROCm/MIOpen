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

/// Reusing the dummy instance of of the ConvolutionContext class
/// to find out if asm kernels are supported and
/// to properly detect version of ROCm.
/// \todo Get rid of this during implementation of #1938 (60)
static auto GetContext(Handle& handle)
{
    ConvolutionContext ctx(conv::Direction::Forward);
    ctx.SetStream(&handle);
    ctx.DetectRocm();
    return ctx;
}

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
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnScale);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnBias);
    }

    const auto resultsave    = resultSaveMean != nullptr && resultSaveInvVariance != nullptr;
    const auto resultrunning = resultRunningMean != nullptr && resultRunningVariance != nullptr;

    const auto problem = batchnorm::ProblemDescription{batchnorm::Direction::ForwardTraining,
                                                       bn_mode,
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
    const auto network_config = problem.MakeNetworkConfig();

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

    if(const auto existingInvoker = handle.GetInvoker(network_config, boost::none, algo))
    {
        (*existingInvoker)(handle, invoke_params);
    }
    else
    {
        const auto ctx = ExecutionContext{&handle};
        const auto solvers =
            solver::SolverContainer<solver::batchnorm::BnFwdTrainingSpatialSingle,
                                    solver::batchnorm::BnFwdTrainingSpatialMultiple,
                                    solver::batchnorm::BnFwdTrainingPerActivation>{};
        const auto slns = solvers.SearchForSolutions(ctx, problem, 1);

        if(slns.empty())
            MIOPEN_THROW(miopenStatusNotImplemented, "No solver found for activation forward.");

        const auto& sln = slns.front();
        if(!sln.invoker_factory)
            MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);
        const auto invoker = handle.PrepareInvoker(*sln.invoker_factory, sln.construction_params);
        handle.RegisterInvoker(invoker, network_config, sln.solver_id, algo);
        invoker(handle, invoke_params);
    }

    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
        miopen::checkNumericsOutput(handle, bnScaleBiasMeanVarDesc, resultRunningMean);
        miopen::checkNumericsOutput(handle, bnScaleBiasMeanVarDesc, resultRunningVariance);
        miopen::checkNumericsOutput(handle, bnScaleBiasMeanVarDesc, resultSaveMean);
        miopen::checkNumericsOutput(handle, bnScaleBiasMeanVarDesc, resultSaveInvVariance);
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
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnScale);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnBias);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, estimatedMean);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, estimatedVariance);
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

        bool bfpmixparm = false;
        bool bfp16parm  = false;
        bool bfp32parm  = true;
        if(xDesc.GetType() == miopenHalf && bnScaleBiasMeanVarDesc.GetType() == miopenHalf)
        {
            bfp16parm = true;
            bfp32parm = false;
        }
        else if(xDesc.GetType() == miopenHalf && bnScaleBiasMeanVarDesc.GetType() == miopenFloat)
        {
            bfpmixparm = true;
            bfp32parm  = false;
        }

        int n, c, h, w;
        std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

        unsigned int in_nstride = c * h * w;
        unsigned int in_cstride = h * w;

        std::string algo_name      = "miopenBatchNormalizationForwardInference";
        std::string network_config = "fp16" + std::to_string(static_cast<int>(bfp16parm)) + "fp32" +
                                     std::to_string(static_cast<int>(bfp32parm)) + "mode" +
                                     std::to_string(bn_mode) + "HWdims" +
                                     std::to_string(in_cstride) + "C" + std::to_string(c);

        auto&& kernels = handle.GetKernels(algo_name, network_config);
        if(!kernels.empty())
        {
            auto kernel = kernels.front();
            kernel(x,
                   y,
                   estimatedMean,
                   estimatedVariance,
                   bnScale,
                   bnBias,
                   epsilon,
                   n,
                   in_cstride,
                   in_nstride);
        }
        else
        {
            size_t xlocalsize = 1;
            auto xgridsize    = c;
            size_t ylocalsize = 256;
            size_t ygridsize  = ylocalsize * ((in_cstride + ylocalsize - 1) / ylocalsize);
            size_t zlocalsize = 1;
            size_t zgridsize  = 1;

            std::string program_name = "MIOpenBatchNormFwdInfer"; // build this up
            std::string kernel_name  = "MIOpenBatchNormFwdInfer";
            if(bn_mode == miopenBNSpatial)
            { // SPATIAL kernels
                program_name += "Spatial.cl";
                kernel_name += "SpatialEst";
            }
            else
            { // PER ACTIVATION
                program_name += "PerAct.cl";
                kernel_name += "PerActivationEst";
            }

            std::string parms =
                " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(bfp16parm)) +
                " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(bfp32parm)) +
                " -DMIOPEN_USE_FPMIX=" + std::to_string(static_cast<int>(bfpmixparm)) +
                " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) +
                " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) +
                " -DMIO_BN_GRP2=" + std::to_string(zlocalsize) +
                " -DMIO_BN_GFX1030=" + ((handle.GetDeviceName() == "gfx1030") ? "1" : "0");

            std::vector<size_t> vld;
            std::vector<size_t> vgd;
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            MIOPEN_LOG_I2(kernel_name << ":: " << parms);

            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x,
                y,
                estimatedMean,
                estimatedVariance,
                bnScale,
                bnBias,
                epsilon,
                n,
                in_cstride,
                in_nstride);
        }
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
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, dyDesc, dy);
        miopen::checkNumericsInput(handle, bnScaleBiasDiffDesc, bnScale);

        miopen::checkNumericsInput(handle, bnScaleBiasDiffDesc, savedMean);
        miopen::checkNumericsInput(handle, bnScaleBiasDiffDesc, savedInvVariance);
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
    const auto network_config = problem.MakeNetworkConfig();

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

    if(const auto existingInvoker = handle.GetInvoker(network_config, boost::none, algo))
    {
        (*existingInvoker)(handle, invoke_params);

        if(miopen::CheckNumericsEnabled())
        {
            miopen::checkNumericsOutput(handle, dxDesc, dx);
            miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnScaleDiff);
            miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnBiasDiff);
        }

        return;
    }
    else
    {
        const auto ctx = ExecutionContext{&handle};
        const auto solvers =
            solver::SolverContainer<solver::batchnorm::BnBwdTrainingSpatialSingle,
                                    solver::batchnorm::BnBwdTrainingSpatialMultiple>{};
        const auto slns = solvers.SearchForSolutions(ctx, problem, 1);

        // if(slns.empty())
        //    MIOPEN_THROW(miopenStatusNotImplemented, "No solver found for activation forward.");

        if(!slns.empty())
        {
            const auto& sln = slns.front();
            if(!sln.invoker_factory)
                MIOPEN_THROW(miopenStatusInternalError,
                             "Invoker missing in solver " + sln.solver_id);
            const auto invoker =
                handle.PrepareInvoker(*sln.invoker_factory, sln.construction_params);
            handle.RegisterInvoker(invoker, network_config, sln.solver_id, algo);
            invoker(handle, invoke_params);

            if(miopen::CheckNumericsEnabled())
            {
                miopen::checkNumericsOutput(handle, dxDesc, dx);
                miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnScaleDiff);
                miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnBiasDiff);
            }

            return;
        }
    }

    static const auto ctx = GetContext(handle);

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    bool bfpmixparm = false;
    bool bfp16parm  = false;
    bool bfp32parm  = true;
    if(xDesc.GetType() == miopenHalf && bnScaleBiasDiffDesc.GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
    }
    else if(xDesc.GetType() == miopenHalf && bnScaleBiasDiffDesc.GetType() == miopenFloat)
    {
        bfpmixparm = true;
        bfp32parm  = false;
    }

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nstride = c * in_cstride;
    unsigned int in_nhw     = n * in_cstride;
    unsigned int in_nchw    = n * in_nstride;

    if(bn_mode == miopenBNSpatial)
    { // SPATIAL kernels
        MIOPEN_THROW(miopenStatusInternalError,
                     "Batchnorm spatial has already been implemented as solver-invoker pairs");
    } // END spatial
    else
    { // PER ACT
        size_t xlocalsize = 1;
        size_t ylocalsize = 1;
        size_t zlocalsize = 1;

        size_t xgridsize = 1;
        size_t ygridsize = 1;
        size_t zgridsize = 1;
        
        ylocalsize           = (64 >= in_cstride) ? 64 : 256;
        unsigned int segment = std::ceil(double(in_cstride) / double(ylocalsize));
        xgridsize            = c;
        ygridsize            = segment * ylocalsize;

        auto&& kernels = handle.GetKernels(algo, network_config);

        if(!kernels.empty())
        {
            auto kernel = kernels.front();

            if(useSaved)
            {
                kernel(x,
                       dy,
                       n,
                       in_nstride,
                       in_cstride,
                       dx,
                       bnScale,
                       resultBnScaleDiff,
                       resultBnBiasDiff,
                       savedMean,
                       savedInvVariance);
            }
            else
            {
                kernel(x,
                       dy,
                       n,
                       in_nstride,
                       in_cstride,
                       dx,
                       bnScale,
                       resultBnScaleDiff,
                       resultBnBiasDiff,
                       epsilon);
            }
        }
        else
        {

            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            std::string program_name = "MIOpenBatchNormBwdPerAct.cl";
            std::string kernel_name  = "MIOpenBatchNormBwdPerActivation";

            std::string parms =
                " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(bfp16parm)) +
                " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(bfp32parm)) +
                " -DMIOPEN_USE_FPMIX=" + std::to_string(static_cast<int>(bfpmixparm)) +
                " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_C=" + std::to_string(c) +
                " -DMIO_BN_HW=" + std::to_string(in_cstride) +
                " -DMIO_BN_NHW=" + std::to_string(in_nhw) +
                " -DMIO_BN_CHW=" + std::to_string(in_nstride) +
                " -DMIO_BN_NCHW=" + std::to_string(in_nchw) +
                " -DMIO_BN_NGRPS=" + std::to_string(int(std::ceil(float(ygridsize) / ylocalsize))) +
                " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) +
                " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) +
                " -DMIO_BN_GRP2=" + std::to_string(zlocalsize) +
                " -DMIO_BN_GFX1030=" + ((handle.GetDeviceName() == "gfx1030") ? "1" : "0");

            if(useSaved)
            {
                kernel_name += "Saved";
                handle.AddKernel(algo, network_config, program_name, kernel_name, vld, vgd, parms)(
                    x,
                    dy,
                    n,
                    in_nstride,
                    in_cstride,
                    dx,
                    bnScale,
                    resultBnScaleDiff,
                    resultBnBiasDiff,
                    savedMean,
                    savedInvVariance);
            }
            else
            {
                handle.AddKernel(algo, network_config, program_name, kernel_name, vld, vgd, parms)(
                    x,
                    dy,
                    n,
                    in_nstride,
                    in_cstride,
                    dx,
                    bnScale,
                    resultBnScaleDiff,
                    resultBnBiasDiff,
                    epsilon);
            }
        }
    }
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
        miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnScaleDiff);
        miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnBiasDiff);
    }
}
} // namespace miopen
