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

#define WORKAROUND_ISSUE_1146 1 // check asm solver applicability for gfx90a

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
        const auto ctx     = ExecutionContext{&handle};
        const auto solvers = solver::SolverContainer<>{};
        const auto slns    = solvers.SearchForSolutions(ctx, problem, 1);

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

    auto inhw = float(1.0 / in_nhw);

    size_t xlocalsize = 1;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;

    size_t xgridsize = 1;
    size_t ygridsize = 1;
    size_t zgridsize = 1;

    if(bn_mode == miopenBNSpatial)
    { // SPATIAL kernels

        unsigned int ldsgcn   = 0;
        unsigned int ldsnogcn = 0;
        bool single           = true;
        int variant           = 1;

        //*************************************************************************************************
        // N*H*W < 32M and H*W > 1024, use batchnorm variant#1 implementation which parallelize
        // work groups over channels and loop through NHW.
        //*************************************************************************************************
        if((in_nhw < (32 * 1024 * 1024) && in_cstride > 1024))
        {
            variant    = 1;
            xlocalsize = 1024;
            xgridsize  = c * xlocalsize;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }
        //*************************************************************************************************
        // N*H*W < 32M and H*W > 512  use batchnorm variant#1 or variant#3 implementation which
        // parallelize
        // work groups over channels and loop through N.
        //*************************************************************************************************
        else if(in_nhw < (32 * 1024 * 1024) && in_cstride > 512)
        {
            variant    = (n >= 32) ? 1 : 3;
            xlocalsize = std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned int>(1024));
            xgridsize  = c * xlocalsize;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }
        //*************************************************************************************************
        // H*W < 512  use batchnorm variant#0 or variant#3 implementation based on batch size and
        // H*W
        //*************************************************************************************************
        else if(in_cstride <= 512)
        {
            if((n > 64) && (in_cstride > 160))
            {
                variant = 3;
                xlocalsize =
                    std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned int>(1024));
                xgridsize = c * xlocalsize;
                ldsgcn    = xlocalsize / 64;
                ldsnogcn  = xlocalsize;
            }
            else
            {
                variant = 0;
                if(bfp32parm)
                {
                    xlocalsize = 1024;
                    xgridsize  = 1024 * c;
                }
                else
                {
                    xlocalsize = 256;
                    xgridsize  = 256 * c;
                }
                ldsgcn   = xlocalsize / 64;
                ldsnogcn = xlocalsize;
            }
        }
        //*************************************************************************************************
        // N*H*W > 32M, use batchnorm variant#2 implementation which parallelize
        // work groups over channels and data segments.
        //*************************************************************************************************
        else
        {
            variant      = 2;
            ylocalsize   = 1024;
            auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize    = c;
            ygridsize    = segment * ylocalsize;
            single       = false;
            ldsgcn       = ylocalsize / 64;
            ldsnogcn     = ylocalsize;
        }
        if((in_cstride < 200) && (in_cstride > 60) && bfpmixparm)
        {
            variant    = 1;
            xlocalsize = 1024;
            xgridsize  = c * xlocalsize;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }

        auto&& kernels = handle.GetKernels(algo, network_config);

        if(single)
        {
            if(!kernels.empty())
            {
                auto kernel = kernels.front();
                visit_float(bnScaleBiasDiffDesc.GetType(), [&](auto as_float) {
                    if(useSaved)
                    {
                        kernel(x,
                               dy,
                               dx,
                               bnScale,
                               resultBnScaleDiff,
                               resultBnBiasDiff,
                               savedMean,
                               savedInvVariance,
                               as_float(inhw));
                    }
                    else
                    {
                        kernel(
                            x, dy, dx, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, inhw);
                    }
                });
            }
            else
            {

                std::string kernel_name;
                std::string program_name;
                std::string parms;

                if((n > 64) && (n % 2 == 0) && (variant == 3) && (bfpmixparm) && (useSaved) &&
                   ctx.use_asm_kernels && ctx.rmv.IsV2orV3() &&
                   (StartsWith(handle.GetDeviceName(), "gfx8") ||
                    (StartsWith(handle.GetDeviceName(), "gfx9")
#if WORKAROUND_ISSUE_1146
                     && (handle.GetDeviceName() != "gfx90a")
#endif
                         )) &&
                   (!handle.GetTargetProperties().Xnack() ||
                    !*handle.GetTargetProperties().Xnack()))
                {
                    kernel_name  = "miopenGcnAsmBNBwdTrainSpatial";
                    program_name = "gcnAsmBNBwdTrainSpatial.s";

                    union nhw_val
                    {
                        unsigned u32;
                        float f32;
                        nhw_val()
                        {
                            u32 = 0;
                            f32 = 0;
                        }
                    } NHW_value;
                    NHW_value.f32 = static_cast<float>(in_nhw);

                    // clang-format off
                    parms = std::string() +
                            " -Wa,-defsym,ROCM_METADATA_VERSION=" + (ctx.rmv.UseV3() ? "5" : "4") +
                            " -Wa,-defsym,MIOPEN_USE_FP16=" + std::to_string(static_cast<int>(bfp16parm)) +
                            " -Wa,-defsym,MIOPEN_USE_FP32=" + std::to_string(static_cast<int>(bfp32parm)) +
                            " -Wa,-defsym,MIOPEN_USE_FPMIX=" + std::to_string(static_cast<int>(bfpmixparm)) +
                            " -Wa,-defsym,MIO_BN_USESAVED=" + std::to_string(static_cast<int>(useSaved)) +
                            " -Wa,-defsym,MIO_BN_N=" + std::to_string(n) +
                            " -Wa,-defsym,MIO_BN_C=" + std::to_string(c) +
                            " -Wa,-defsym,MIO_BN_HW=" + std::to_string(in_cstride) +
                            " -Wa,-defsym,MIO_BN_NHW=" + std::to_string(in_nhw) +
                            " -Wa,-defsym,MIO_BN_NHW_FLOAT=" + std::to_string(NHW_value.u32) +
                            " -Wa,-defsym,MIO_BN_CHW=" + std::to_string(in_nstride) +
                            " -Wa,-defsym,MIO_BN_NCHW=" + std::to_string(in_nchw) +
                            " -Wa,-defsym,MIO_BN_LDS_SIZE=" + std::to_string(ldsnogcn) +
                            " -Wa,-defsym,MIO_BN_LDSGCN_SIZE=" + std::to_string(ldsgcn) +
                            " -Wa,-defsym,MIO_BN_VARIANT=" + std::to_string(variant) +
                            " -Wa,-defsym,MIO_BN_GRP0=" + std::to_string(xlocalsize) +
                            " -Wa,-defsym,MIO_BN_GRP1=" + std::to_string(ylocalsize) +
                            " -Wa,-defsym,MIO_BN_GRP2=" + std::to_string(zlocalsize); // clang-format on
                }
                else
                {
                    program_name = "MIOpenBatchNormBwdSpatial.cl";
                    kernel_name  = "MIOpenBatchNormBwdSpatial";

                    parms =
                        " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(bfp16parm)) +
                        " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(bfp32parm)) +
                        " -DMIOPEN_USE_FPMIX=" + std::to_string(static_cast<int>(bfpmixparm)) +
                        " -DMIO_BN_USESAVED=" + std::to_string(static_cast<int>(useSaved)) +
                        " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_C=" + std::to_string(c) +
                        " -DMIO_BN_HW=" + std::to_string(in_cstride) +
                        " -DMIO_BN_NHW=" + std::to_string(in_nhw) +
                        " -DMIO_BN_CHW=" + std::to_string(in_nstride) +
                        " -DMIO_BN_NCHW=" + std::to_string(in_nchw) +
                        " -DMIO_BN_LDS_SIZE=" + std::to_string(ldsnogcn) +
                        " -DMIO_BN_LDSGCN_SIZE=" + std::to_string(ldsgcn) +
                        " -DMIO_BN_VARIANT=" + std::to_string(variant) +
                        " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) +
                        " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) +
                        " -DMIO_BN_GRP2=" + std::to_string(zlocalsize) +
                        " -DMIO_BN_GFX1030=" + ((handle.GetDeviceName() == "gfx1030") ? "1" : "0");
                }

                MIOPEN_LOG_I2(kernel_name << ":: " << algo.ToString());
                MIOPEN_LOG_I2("..." << parms);
                MIOPEN_LOG_I2("..." << network_config.ToString());
                vld.push_back(xlocalsize);
                vld.push_back(ylocalsize);
                vld.push_back(zlocalsize);

                vgd.push_back(xgridsize);
                vgd.push_back(ygridsize);
                vgd.push_back(zgridsize);

                MIOPEN_LOG_I2(kernel_name << ":: " << parms);

                bnBwdTrainSelectSingle(handle,
                                       bnScaleBiasDiffDesc.GetType(),
                                       program_name,
                                       algo,
                                       kernel_name,
                                       network_config,
                                       parms,
                                       vld,
                                       vgd,
                                       x,
                                       dy,
                                       dx,
                                       bnScale,
                                       resultBnScaleDiff,
                                       resultBnBiasDiff,
                                       useSaved,
                                       epsilon,
                                       savedMean,
                                       savedInvVariance,
                                       inhw);
            }
        }
        else // Use multi-kernel
        {
            if(!kernels.empty())
            {
                float ctime = 0.;
                visit_float(bnScaleBiasDiffDesc.GetType(), [&](auto as_float) {
                    if(useSaved)
                    {
                        kernels[0](x, dy, dx, savedMean, savedInvVariance);
                        profileSequence(handle, 0, &ctime);

                        kernels[1](dx, resultBnScaleDiff, resultBnBiasDiff);
                        profileSequence(handle, 1, &ctime);

                        kernels[2](x,
                                   dy,
                                   dx,
                                   bnScale,
                                   resultBnScaleDiff,
                                   resultBnBiasDiff,
                                   savedMean,
                                   savedInvVariance,
                                   as_float(inhw));
                        profileSequence(handle, 2, &ctime);
                    }
                    else
                    {
                        kernels[0](x, dx); // mean variance
                        profileSequence(handle, 0, &ctime);

                        kernels[1](dx, as_float(inhw), epsilon); // final mean variance
                        profileSequence(handle, 1, &ctime);

                        kernels[2](x, dy, dx); // dscale dbias
                        profileSequence(handle, 1, &ctime);

                        kernels[3](dx, resultBnScaleDiff, resultBnBiasDiff); // final dscale dbias
                        profileSequence(handle, 1, &ctime);

                        kernels[4](x,
                                   dy,
                                   dx,
                                   bnScale,
                                   resultBnScaleDiff,
                                   resultBnBiasDiff,
                                   as_float(inhw)); // dx
                        profileSequence(handle, 2, &ctime);
                    }
                });
            }
            else
            {

                vld.push_back(xlocalsize);
                vld.push_back(ylocalsize);
                vld.push_back(zlocalsize);

                vgd.push_back(xgridsize);
                vgd.push_back(ygridsize);
                vgd.push_back(zgridsize);

                std::string program_name = "MIOpenBatchNormBwdSpatial.cl";
                std::string kernel_name  = "MIOpenBatchNormBwdSpatial";
                std::string parms =
                    " -DMIOPEN_USE_FP16=" + std::to_string(static_cast<int>(bfp16parm)) +
                    " -DMIOPEN_USE_FP32=" + std::to_string(static_cast<int>(bfp32parm)) +
                    " -DMIOPEN_USE_FPMIX=" + std::to_string(static_cast<int>(bfpmixparm)) +
                    " -DMIO_BN_USESAVED=" + std::to_string(static_cast<int>(useSaved)) +
                    " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_C=" + std::to_string(c) +
                    " -DMIO_BN_HW=" + std::to_string(in_cstride) +
                    " -DMIO_BN_NHW=" + std::to_string(in_nhw) +
                    " -DMIO_BN_CHW=" + std::to_string(in_nstride) +
                    " -DMIO_BN_NCHW=" + std::to_string(in_nchw) + " -DMIO_BN_NGRPS=" +
                    std::to_string(int(std::ceil(float(ygridsize) / ylocalsize))) +
                    " -DMIO_BN_LDS_SIZE=" + std::to_string(ldsnogcn) +
                    " -DMIO_BN_LDSGCN_SIZE=" + std::to_string(ldsgcn) +
                    " -DMIO_BN_VARIANT=" + std::to_string(variant) +
                    " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) +
                    " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) +
                    " -DMIO_BN_GRP2=" + std::to_string(zlocalsize) +
                    " -DMIO_BN_GFX1030=" + ((handle.GetDeviceName() == "gfx1030") ? "1" : "0");

                MIOPEN_LOG_I2(kernel_name << ":: " << parms);

                bnBwdTrainSelectMulti(handle,
                                      bnScaleBiasDiffDesc.GetType(),
                                      program_name,
                                      algo,
                                      kernel_name,
                                      network_config,
                                      parms,
                                      vld,
                                      vgd,
                                      x,
                                      dy,
                                      dx,
                                      bnScale,
                                      resultBnScaleDiff,
                                      resultBnBiasDiff,
                                      useSaved,
                                      epsilon,
                                      savedMean,
                                      savedInvVariance,
                                      inhw);
            }
        }
    } // END spatial
    else
    { // PER ACT

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
