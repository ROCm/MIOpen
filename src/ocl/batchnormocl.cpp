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
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/visit_float.hpp>
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
    if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != bnScaleBiasMeanVarDesc.GetType())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!xDesc.IsPacked())
    {
        std::cerr << "Only fully packed tensors supported." << std::endl;
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

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nstride = c * in_cstride;
    unsigned int in_nhw     = n * in_cstride;
    unsigned int in_nchw    = n * in_nstride;

    size_t xlocalsize = 1;
    size_t ylocalsize = 1;
    size_t zlocalsize = 1;

    size_t xgridsize = 1;
    size_t ygridsize = 1;
    size_t zgridsize = 1;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    bool bfp16parm = false;
    bool bfp32parm = true;
    if(xDesc.GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
    }

    bool resultsave = false;
    if(resultSaveMean != nullptr && resultSaveInvVariance != nullptr)
    {
        resultsave = true;
    }

    bool resultrunning = false;
    if(resultRunningMean != nullptr && resultRunningVariance != nullptr)
    {
        resultrunning = true;
    }
    auto inhw = float(1.0 / in_nhw);

    if(bn_mode == miopenBNSpatial)
    {
        bool single           = true;
        unsigned int variant  = 1;
        unsigned int ldsgcn   = 0;
        unsigned int ldsnogcn = 0;
        if(in_nhw < 33554432 && in_cstride > 1024)
        {
            variant    = 1;
            ylocalsize = std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned int>(1024));
            xgridsize  = c;
            ygridsize  = ylocalsize;
            ldsgcn     = ylocalsize / 64;
            ldsnogcn   = ylocalsize;
        }
        else if(in_nhw < 33554432 && in_cstride > 512)
        {
            variant    = 3;
            ylocalsize = 64 * ((in_cstride + 63) / 64);
            xgridsize  = c;
            ygridsize  = ylocalsize;
            ldsgcn     = ylocalsize / 64;
            ldsnogcn   = ylocalsize;
        }
        else if(in_cstride <= 512)
        {
            xlocalsize = 1024;
            variant    = 0;
            xgridsize  = xlocalsize * c;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }
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

        std::string algo_name = "miopenBatchNormForwardTrainingSpatial";
        std::string network_config =
            std::to_string(variant) + std::to_string(xgridsize) + std::to_string(ldsgcn) +
            std::to_string(ygridsize) + std::to_string(xlocalsize) + std::to_string(ylocalsize) +
            "rs" + std::to_string(resultsave) + std::to_string(resultrunning) + "type" +
            std::to_string(bfp16parm) + std::to_string(bfp32parm) + std::to_string(in_nchw) +
            std::to_string(single) + std::to_string(in_cstride);

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(single)
        {
            if(!kernels.empty())
            {
                auto kernel = kernels.front();
                visit_float(xDesc.GetType(), [&](auto as_float) {
                    if(resultsave && resultrunning)
                    {

                        kernel(x,
                               y,
                               bnScale,
                               bnBias,
                               as_float(inhw),
                               expAvgFactor,
                               resultRunningMean,
                               resultRunningVariance,
                               epsilon,
                               resultSaveMean,
                               resultSaveInvVariance);
                    }
                    else if(resultsave)
                    {
                        kernel(x,
                               y,
                               bnScale,
                               bnBias,
                               as_float(inhw),
                               epsilon,
                               resultSaveMean,
                               resultSaveInvVariance);
                    }
                    else if(resultrunning)
                    {
                        kernel(x,
                               y,
                               bnScale,
                               bnBias,
                               as_float(inhw),
                               expAvgFactor,
                               resultRunningMean,
                               resultRunningVariance,
                               epsilon);
                    }
                    else
                    {
                        kernel(x, y, bnScale, bnBias, as_float(inhw), epsilon);
                    }
                });
            }
            else
            {

                std::string kernel_name  = "BatchNormFwdTrainSpatial";
                std::string program_name = "MIOpenBatchNormFwdTrainSpatial.cl";

                vld.push_back(xlocalsize);
                vld.push_back(ylocalsize);
                vld.push_back(zlocalsize);

                vgd.push_back(xgridsize);
                vgd.push_back(ygridsize);
                vgd.push_back(zgridsize);

                std::string parms =
                    " -DMIOPEN_USE_FP16=" + std::to_string(bfp16parm) + " -DMIOPEN_USE_FP32=" +
                    std::to_string(bfp32parm) + " -DMIO_SAVE_MEAN_VARIANCE=" +
                    std::to_string(resultsave) + " -DMIO_RUNNING_RESULT=" +
                    std::to_string(resultrunning) + " -DMIO_BN_N=" + std::to_string(n) +
                    " -DMIO_BN_C=" + std::to_string(c) + " -DMIO_BN_HW=" +
                    std::to_string(in_cstride) + " -DMIO_BN_NHW=" + std::to_string(in_nhw) +
                    " -DMIO_BN_CHW=" + std::to_string(in_nstride) + " -DMIO_BN_NCHW=" +
                    std::to_string(in_nchw) + " -DMIO_BN_LDS_SIZE=" + std::to_string(ldsnogcn) +
                    " -DMIO_BN_LDSGCN_SIZE=" + std::to_string(ldsgcn) + " -DMIO_BN_VARIANT=" +
                    std::to_string(variant) + " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) +
                    " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) + " -DMIO_BN_GRP2=" +
                    std::to_string(zlocalsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << kernel_name << ":: ";
                std::cout << algo_name << std::endl;
                std::cout << parms << std::endl;
                std::cout << network_config << std::endl;
#endif
                bnFwdTrainSelectSingle(handle,
                                       xDesc.GetType(),
                                       program_name,
                                       algo_name,
                                       kernel_name,
                                       network_config,
                                       parms,
                                       vld,
                                       vgd,
                                       x,
                                       y,
                                       bnScale,
                                       bnBias,
                                       resultsave,
                                       resultrunning,
                                       expAvgFactor,
                                       resultRunningMean,
                                       resultRunningVariance,
                                       epsilon,
                                       resultSaveMean,
                                       resultSaveInvVariance,
                                       inhw);
            }
        }
        else
        {
            if(!kernels.empty())
            {
                float ctime = 0.;
                visit_float(xDesc.GetType(), [&](auto as_float) {
                    if(resultsave && resultrunning)
                    {
                        kernels[0](x, y);
                        profileSequence(handle, 0, &ctime);

                        kernels[1](y,
                                   as_float(inhw),
                                   expAvgFactor,
                                   resultRunningMean,
                                   resultRunningVariance,
                                   epsilon,
                                   resultSaveMean,
                                   resultSaveInvVariance);
                        profileSequence(handle, 1, &ctime);

                        kernels[2](x, y, bnScale, bnBias);
                        profileSequence(handle, 2, &ctime);
                    }
                    else if(resultsave)
                    {
                        kernels[0](x, y);
                        profileSequence(handle, 0, &ctime);

                        kernels[1](
                            y, as_float(inhw), epsilon, resultSaveMean, resultSaveInvVariance);
                        profileSequence(handle, 1, &ctime);

                        kernels[2](x, y, bnScale, bnBias);
                        profileSequence(handle, 2, &ctime);
                    }
                    else if(resultrunning)
                    {

                        kernels[0](x, y);
                        profileSequence(handle, 0, &ctime);

                        kernels[1](y,
                                   as_float(inhw),
                                   expAvgFactor,
                                   resultRunningMean,
                                   resultRunningVariance,
                                   epsilon);
                        profileSequence(handle, 1, &ctime);

                        kernels[2](x, y, bnScale, bnBias);
                        profileSequence(handle, 2, &ctime);
                    }
                    else
                    {
                        kernels[0](x, y);
                        profileSequence(handle, 0, &ctime);

                        kernels[1](y, as_float(inhw), epsilon);
                        profileSequence(handle, 1, &ctime);

                        kernels[2](x, y, bnScale, bnBias);
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

                std::string kernel_name  = "BatchNormFwdTrainSpatial";
                std::string program_name = "MIOpenBatchNormFwdTrainSpatial.cl";
                std::string parms =
                    " -DMIOPEN_USE_FP16=" + std::to_string(bfp16parm) + " -DMIOPEN_USE_FP32=" +
                    std::to_string(bfp32parm) + " -DMIO_SAVE_MEAN_VARIANCE=" +
                    std::to_string(resultsave) + " -DMIO_RUNNING_RESULT=" +
                    std::to_string(resultrunning) + " -DMIO_BN_N=" + std::to_string(n) +
                    " -DMIO_BN_C=" + std::to_string(c) + " -DMIO_BN_HW=" +
                    std::to_string(in_cstride) + " -DMIO_BN_NHW=" + std::to_string(in_nhw) +
                    " -DMIO_BN_CHW=" + std::to_string(in_nstride) + " -DMIO_BN_NCHW=" +
                    std::to_string(in_nchw) + " -DMIO_BN_NGRPS=" +
                    std::to_string(int(std::ceil(float(ygridsize) / ylocalsize))) +
                    " -DMIO_BN_LDS_SIZE=" + std::to_string(ldsnogcn) + " -DMIO_BN_LDSGCN_SIZE=" +
                    std::to_string(ldsgcn) + " -DMIO_BN_VARIANT=" + std::to_string(variant) +
                    " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) + " -DMIO_BN_GRP1=" +
                    std::to_string(ylocalsize) + " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << kernel_name << ":: ";
                std::cout << parms << std::endl;
#endif

                bnFwdTrainSelectMulti(handle,
                                      xDesc.GetType(),
                                      program_name,
                                      algo_name,
                                      kernel_name,
                                      network_config,
                                      parms,
                                      vld,
                                      vgd,
                                      x,
                                      y,
                                      bnScale,
                                      bnBias,
                                      resultsave,
                                      resultrunning,
                                      expAvgFactor,
                                      resultRunningMean,
                                      resultRunningVariance,
                                      epsilon,
                                      resultSaveMean,
                                      resultSaveInvVariance,
                                      inhw);
            }
        }
    }
    else // else run per activation
    {

        ylocalsize            = 256;
        auto segment          = std::ceil(double(in_cstride) / double(ylocalsize));
        xgridsize             = c;
        ygridsize             = segment * ylocalsize;
        std::string algo_name = "miopenBatchNormForwardTrainingPerActivation";
        std::string network_config =
            std::to_string(bfp16parm) + std::to_string(bfp32parm) + std::to_string(xgridsize) +
            std::to_string(ygridsize) + std::to_string(xlocalsize) + std::to_string(ylocalsize) +
            std::to_string(resultsave) + std::to_string(resultrunning) + std::to_string(in_nchw) +
            std::to_string(segment) + std::to_string(n) + std::to_string(in_cstride);

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(!kernels.empty())
        {
            auto kernel = kernels.front();
            if(resultsave && resultrunning)
            {
                kernel(x,
                       in_nstride,
                       in_cstride,
                       y,
                       bnScale,
                       bnBias,
                       expAvgFactor,
                       resultRunningMean,
                       resultRunningVariance,
                       epsilon,
                       resultSaveMean,
                       resultSaveInvVariance);
            }
            else if(resultsave)
            {
                kernel(x,
                       in_nstride,
                       in_cstride,
                       y,
                       bnScale,
                       bnBias,
                       epsilon,
                       resultSaveMean,
                       resultSaveInvVariance);
            }
            else if(resultrunning)
            {
                kernel(x,
                       in_nstride,
                       in_cstride,
                       y,
                       bnScale,
                       bnBias,
                       expAvgFactor,
                       resultRunningMean,
                       resultRunningVariance,
                       epsilon);
            }
            else
            {
                kernel(x, in_nstride, in_cstride, y, bnScale, bnBias, epsilon);
            }
        }
        else // kernels empty
        {

            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            std::string parms =
                " -DMIOPEN_USE_FP16=" + std::to_string(bfp16parm) + " -DMIOPEN_USE_FP32=" +
                std::to_string(bfp32parm) + " -DMIO_SAVE_MEAN_VARIANCE=" +
                std::to_string(resultsave) + " -DMIO_RUNNING_RESULT=" +
                std::to_string(resultrunning) + " -DMIO_BN_N=" + std::to_string(n) +
                " -DMIO_BN_C=" + std::to_string(c) + " -DMIO_BN_HW=" + std::to_string(in_cstride) +
                " -DMIO_BN_NHW=" + std::to_string(in_nhw) + " -DMIO_BN_CHW=" +
                std::to_string(in_nstride) + " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize) +
                " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) + " -DMIO_BN_GRP1=" +
                std::to_string(ylocalsize) + " -DMIO_BN_GRP2=" + std::to_string(zlocalsize) +
                " -DMIO_BN_NCHW=" + std::to_string(in_nchw);

            std::string program_name = "MIOpenBatchNormFwdTrainPerAct.cl";
            std::string kernel_name  = "BatchNormFwdTrainPerActivation";

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << kernel_name << ":: ";
            std::cout << parms << std::endl;
            printf("No kernel found, adding kernel.\nxgridsize: %ld, ygridsize: %ld\n",
                   xgridsize,
                   ygridsize);
#endif
            if(resultsave && resultrunning)
            {
                handle.AddKernel(
                    algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                    x,
                    in_nstride,
                    in_cstride,
                    y,
                    bnScale,
                    bnBias,
                    expAvgFactor,
                    resultRunningMean,
                    resultRunningVariance,
                    epsilon,
                    resultSaveMean,
                    resultSaveInvVariance);
            }
            else if(resultsave)
            {
                handle.AddKernel(
                    algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                    x,
                    in_nstride,
                    in_cstride,
                    y,
                    bnScale,
                    bnBias,
                    epsilon,
                    resultSaveMean,
                    resultSaveInvVariance);
            }
            else if(resultrunning)
            {
                handle.AddKernel(
                    algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                    x,
                    in_nstride,
                    in_cstride,
                    y,
                    bnScale,
                    bnBias,
                    expAvgFactor,
                    resultRunningMean,
                    resultRunningVariance,
                    epsilon);
            }
            else
            {
                handle.AddKernel(
                    algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                    x, in_nstride, in_cstride, y, bnScale, bnBias, epsilon);
            }
        }
    } // end per-activation

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
        if(xDesc.GetType() != yDesc.GetType() ||
           xDesc.GetType() != bnScaleBiasMeanVarDesc.GetType())
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
            std::cerr << "Only alpha=1 and beta=0 is supported" << std::endl;
            MIOPEN_THROW(miopenStatusBadParm);
        }

        bool bfp16parm = false;
        bool bfp32parm = true;
        if(xDesc.GetType() == miopenHalf)
        {
            bfp16parm = true;
            bfp32parm = false;
        }

        int n, c, h, w;
        std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

        unsigned int in_nstride = c * h * w;
        unsigned int in_cstride = h * w;

        size_t xlocalsize = 1;
        auto ylocalsize   = size_t((in_cstride > 1024) ? 1024 : ((64 >= in_cstride) ? 64 : 256));

        std::vector<size_t> vld;
        std::vector<size_t> vgd;

        auto segment   = std::ceil(double(in_cstride) / double(ylocalsize));
        auto xgridsize = size_t(c);
        auto ygridsize = size_t(segment * ylocalsize);

        std::string algo_name      = "miopenBatchNormalizationForwardInference";
        std::string network_config = std::to_string(n) + std::to_string(in_cstride) +
                                     std::to_string(in_nstride) + std::to_string(segment) + "dims" +
                                     std::to_string(xgridsize) + std::to_string(ygridsize) +
                                     std::to_string(xlocalsize) + std::to_string(ylocalsize) +
                                     +"type" + std::to_string(bfp16parm) +
                                     std::to_string(bfp32parm) + "mode" + std::to_string(bn_mode);

        auto&& kernels = handle.GetKernels(algo_name, network_config);
        if(!kernels.empty())
        {
            auto kernel = kernels.front();
            kernel(x, y, estimatedMean, estimatedVariance, bnScale, bnBias, epsilon);
        }
        else
        {

            size_t zlocalsize        = 1;
            size_t zgridsize         = 1;
            std::string program_name = "MIOpenBatchNormFwdInfer"; // build this up
            std::string kernel_name  = "BatchNormFwdInfer";
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
                " -DMIOPEN_USE_FP16=" + std::to_string(bfp16parm) + " -DMIOPEN_USE_FP32=" +
                std::to_string(bfp32parm) + " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_HW=" +
                std::to_string(in_cstride) + " -DMIO_BN_CHW=" + std::to_string(in_nstride) +
                " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) + " -DMIO_BN_GRP1=" +
                std::to_string(ylocalsize) + " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);

            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << kernel_name << ":: ";
            std::cout << parms << std::endl;
#endif
            handle.AddKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x, y, estimatedMean, estimatedVariance, bnScale, bnBias, epsilon);
        }
    }
    else // Need to recalculated everything, let's just call training kernel in that case
    {

#if(MIOPEN_BN_CPP_DEBUG == 1)
        std::cout << "Call to fwd train from forward inference:: ";
#endif
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
    if(dxDesc.GetType() != dyDesc.GetType() || dyDesc.GetType() != xDesc.GetType() ||
       xDesc.GetType() != bnScaleBiasDiffDesc.GetType())
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
        std::cerr << "Only alphaDataDiff=1 and betaDataDiff=0 is supported" << std::endl;
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(!float_equal(*(static_cast<const float*>(alphaParamDiff)), 1.0) ||
       !float_equal(*(static_cast<const float*>(betaParamDiff)), 0))
    {
        std::cerr << "Only alphaParamDiff=1 and betaParamDiff=0 is supported" << std::endl;
        MIOPEN_THROW(miopenStatusBadParm);
    }

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    bool bfp16parm = false;
    bool bfp32parm = true;
    if(xDesc.GetType() == miopenHalf)
    {
        bfp16parm = true;
        bfp32parm = false;
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

    bool useSaved = false;
    if(savedMean != nullptr && savedInvVariance != nullptr)
    {
        useSaved = true;
    }

    if(bn_mode == miopenBNSpatial)
    { // SPATIAL kernels

        unsigned int ldsgcn   = 0;
        unsigned int ldsnogcn = 0;
        bool single           = true;
        unsigned int variant  = 1;

        if(in_nhw < 33554432 && in_cstride > 1024)
        {
            variant    = 1;
            ylocalsize = std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned int>(1024));
            xgridsize  = c;
            ygridsize  = ylocalsize;
            ldsgcn     = ylocalsize / 64;
            ldsnogcn   = ylocalsize;
        }
        else if(in_nhw < 33554432 && in_cstride > 512)
        {
            variant    = 3;
            ylocalsize = std::min(64 * ((in_cstride + 63) / 64), static_cast<unsigned int>(1024));
            xgridsize  = c;
            ygridsize  = ylocalsize;
            ldsgcn     = ylocalsize / 64;
            ldsnogcn   = ylocalsize;
        }
        else if(in_cstride <= 512)
        {
            variant    = 0;
            xlocalsize = 1024;
            xgridsize  = 1024 * c;
            ldsgcn     = xlocalsize / 64;
            ldsnogcn   = xlocalsize;
        }
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

        std::string algo_name = "miopenBatchNormBackwardPropSpatial";
        std::string network_config =
            std::to_string(variant) + std::to_string(xgridsize) + std::to_string(in_cstride) +
            std::to_string(ygridsize) + std::to_string(xlocalsize) + std::to_string(ylocalsize) +
            std::to_string(useSaved) + std::to_string(bfp16parm) + std::to_string(bfp32parm) +
            std::to_string(in_nchw) + std::to_string(single) + std::to_string(c) +
            std::to_string(ldsgcn);

        auto&& kernels = handle.GetKernels(algo_name, network_config);

        if(single)
        {
            if(!kernels.empty())
            {
                auto kernel = kernels.front();
                visit_float(xDesc.GetType(), [&](auto as_float) {
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

                std::string program_name = "MIOpenBatchNormBwdSpatial.cl";
                std::string kernel_name  = "BatchNormBwdSpatial";

                vld.push_back(xlocalsize);
                vld.push_back(ylocalsize);
                vld.push_back(zlocalsize);

                vgd.push_back(xgridsize);
                vgd.push_back(ygridsize);
                vgd.push_back(zgridsize);

                std::string parms =
                    " -DMIOPEN_USE_FP16=" + std::to_string(bfp16parm) + " -DMIOPEN_USE_FP32=" +
                    std::to_string(bfp32parm) + " -DMIO_BN_USESAVED=" + std::to_string(useSaved) +
                    " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_C=" + std::to_string(c) +
                    " -DMIO_BN_HW=" + std::to_string(in_cstride) + " -DMIO_BN_NHW=" +
                    std::to_string(in_nhw) + " -DMIO_BN_CHW=" + std::to_string(in_nstride) +
                    " -DMIO_BN_NCHW=" + std::to_string(in_nchw) + " -DMIO_BN_LDS_SIZE=" +
                    std::to_string(ldsnogcn) + " -DMIO_BN_LDSGCN_SIZE=" + std::to_string(ldsgcn) +
                    " -DMIO_BN_VARIANT=" + std::to_string(variant) + " -DMIO_BN_GRP0=" +
                    std::to_string(xlocalsize) + " -DMIO_BN_GRP1=" + std::to_string(ylocalsize) +
                    " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << kernel_name << ":: ";
                std::cout << parms << std::endl;
#endif

                bnBwdTrainSelectSingle(handle,
                                       xDesc.GetType(),
                                       program_name,
                                       algo_name,
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
                visit_float(xDesc.GetType(), [&](auto as_float) {
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
                std::string kernel_name  = "BatchNormBwdSpatial";
                std::string parms =
                    " -DMIOPEN_USE_FP16=" + std::to_string(bfp16parm) + " -DMIOPEN_USE_FP32=" +
                    std::to_string(bfp32parm) + " -DMIO_BN_USESAVED=" + std::to_string(useSaved) +
                    " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_C=" + std::to_string(c) +
                    " -DMIO_BN_HW=" + std::to_string(in_cstride) + " -DMIO_BN_NHW=" +
                    std::to_string(in_nhw) + " -DMIO_BN_CHW=" + std::to_string(in_nstride) +
                    " -DMIO_BN_NCHW=" + std::to_string(in_nchw) + " -DMIO_BN_NGRPS=" +
                    std::to_string(int(std::ceil(float(ygridsize) / ylocalsize))) +
                    " -DMIO_BN_LDS_SIZE=" + std::to_string(ldsnogcn) + " -DMIO_BN_LDSGCN_SIZE=" +
                    std::to_string(ldsgcn) + " -DMIO_BN_VARIANT=" + std::to_string(variant) +
                    " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) + " -DMIO_BN_GRP1=" +
                    std::to_string(ylocalsize) + " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << kernel_name << ":: ";
                std::cout << parms << std::endl;
#endif

                bnBwdTrainSelectMulti(handle,
                                      xDesc.GetType(),
                                      program_name,
                                      algo_name,
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

        if(savedMean == nullptr || savedInvVariance == nullptr)
        {
            useSaved = false;
        }

        std::string algo_name      = "miopenBatchNormBackwardPropPerActivation";
        std::string network_config = std::to_string(xDesc.GetType()) + std::to_string(xgridsize) +
                                     std::to_string(ygridsize) + std::to_string(xlocalsize) +
                                     std::to_string(ylocalsize) + std::to_string(useSaved) +
                                     std::to_string(bfp16parm) + std::to_string(bfp32parm) +
                                     std::to_string(in_nhw);

        auto&& kernels = handle.GetKernels(algo_name, network_config);

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
            std::string kernel_name  = "BatchNormBwdPerActivation";

            std::string parms =
                " -DMIOPEN_USE_FP16=" + std::to_string(bfp16parm) + " -DMIOPEN_USE_FP32=" +
                std::to_string(bfp32parm) + " -DMIO_BN_N=" + std::to_string(n) + " -DMIO_BN_C=" +
                std::to_string(c) + " -DMIO_BN_HW=" + std::to_string(in_cstride) +
                " -DMIO_BN_NHW=" + std::to_string(in_nhw) + " -DMIO_BN_CHW=" +
                std::to_string(in_nstride) + " -DMIO_BN_NCHW=" + std::to_string(in_nchw) +
                " -DMIO_BN_NGRPS=" + std::to_string(int(std::ceil(float(ygridsize) / ylocalsize))) +
                " -DMIO_BN_GRP0=" + std::to_string(xlocalsize) + " -DMIO_BN_GRP1=" +
                std::to_string(ylocalsize) + " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);

            if(useSaved)
            {
                kernel_name += "Saved";
                handle.AddKernel(
                    algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
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
                handle.AddKernel(
                    algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
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
