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
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        // miopen::checkNumericsInput(handle, yDesc, y); // if beta!=0?
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnScale);
        miopen::checkNumericsInput(handle, bnScaleBiasMeanVarDesc, bnBias);
    }

    std::string program_name = "MIOpenBatchNormFwdTrain";
    std::string algo_name    = "miopenBatchNormalizationForwardTraining";
    std::string kernel_name  = "BatchNormFwdTrain";
    std::string network_config{};

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nstride = c * in_cstride;
    unsigned int in_nhw     = n * in_cstride;
    unsigned int in_nchw    = n * in_nstride;

    size_t xlocalsize = 0;
    size_t ylocalsize = 0;
    size_t zlocalsize = 0;

    size_t xgridsize = 0;
    size_t ygridsize = 0;
    size_t zgridsize = 0;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    // compile parameters
    std::string parms;
    bool resultsave = false;
    if(resultSaveMean != nullptr && resultSaveInvVariance != nullptr)
    {
        parms += "-DMIO_SAVE_MEAN_VARIANCE=1 ";
        resultsave = true;
    }
    else
    {
        parms += "-DMIO_SAVE_MEAN_VARIANCE=0 ";
    }

    bool resultrunning = false;
    if(resultRunningMean != nullptr && resultRunningVariance != nullptr)
    {
        resultrunning = true;
        parms += "-DMIO_RUNNING_RESULT=1 ";
    }
    else
    {
        parms += "-DMIO_RUNNING_RESULT=0 ";
    }

    parms += "-DMIO_BN_N=" + std::to_string(n);
    parms += " -DMIO_BN_C=" + std::to_string(c);
    parms += " -DMIO_BN_HW=" + std::to_string(in_cstride);
    parms += " -DMIO_BN_NHW=" + std::to_string(in_nhw);
    parms += " -DMIO_BN_CHW=" + std::to_string(in_nstride);
    parms += " -DMIO_BN_NCHW=" + std::to_string(in_nchw);

    auto inhw = float(1.0 / in_nhw);

    if(bn_mode == miopenBNSpatial)
    {

        program_name += "Spatial.cl";
        kernel_name += "Spatial";

        if(in_cstride > 1024 && in_nhw < 33554432)
        {
            // unsigned int variant = (in_cstride < 2097152)? 5: 6;
            unsigned int variant = (h == w) ? 5 : 6;

            xlocalsize = 1024;
            ylocalsize = 1;
            zlocalsize = 1;

            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_VARIANT=" + std::to_string(variant);
            parms += " -DMIO_BN_GRP0=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            xgridsize = xlocalsize * c;
            ygridsize = 1;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << kernel_name << ":: ";
            std::cout << parms << std::endl;
            std::cout << "in_nhw: "
                      << ":: " << in_nhw << std::endl;
            std::cout << "inhw: "
                      << ":: " << inhw << std::endl;
#endif
            bnFwdTrainSelectSingle(handle,
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
        else if(in_cstride <= 512 && n > 3 && in_cstride > 4)
        {
            xlocalsize = 1024;
            ylocalsize = 1;
            zlocalsize = 1;

            unsigned int variant = 255;
            unsigned int segment = in_cstride * (xlocalsize / in_cstride);
            unsigned int nloops  = (in_nhw + segment - 1) / segment;

            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_VARIANT=" + std::to_string(variant);
            parms += " -DMIO_BN_GRP0=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);
            parms += " -DMIO_BN_NLOOP=" + std::to_string(nloops);
            parms += " -DMIO_BN_SEGMENT=" + std::to_string((segment > in_nhw) ? in_nhw : segment);
            parms += " -DMIO_BN_SEGIHW=" + std::to_string(segment / in_cstride);
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            xgridsize = 1024 * c;
            ygridsize = 1;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << kernel_name << ":: ";
            std::cout << parms << std::endl;
            std::cout << "in_nhw: "
                      << ":: " << in_nhw << std::endl;
            std::cout << "inhw: "
                      << ":: " << inhw << std::endl;
#endif
            bnFwdTrainSelectSingle(handle,
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
        else if(in_cstride > 1024)
        {
            unsigned int variant = 3;
            xlocalsize           = 1;
            ylocalsize           = 1024;
            zlocalsize           = 1;
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize    = c;
            ygridsize    = segment * ylocalsize;
            zgridsize    = 1;

            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            unsigned int numwgs = std::ceil(float(ygridsize) / ylocalsize);
            parms += " -DMIO_BN_NGRPS=" + std::to_string(numwgs);
            parms += " -DMIO_BN_VARIANT=" + std::to_string(variant);
            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP0=" + std::to_string(1);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(1);
            parms += " -DMIO_BN_SEGMENT=" + std::to_string(segment);

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << kernel_name << ":: ";
            std::cout << parms << std::endl;
#endif
            bnFwdTrainSelectMulti(handle,
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
        else
        {

            xlocalsize = 1;
            zlocalsize = 1;

            unsigned int variant;

            if(in_cstride < 257 && in_cstride > n && n <= 64 && in_cstride > 1)
            {
                variant    = 0;
                ylocalsize = (in_cstride <= 16) ? 16 : ((in_cstride <= 64) ? 64 : 256);
            }
            else if(in_cstride <= 64)
            {
                variant    = 1;
                ylocalsize = (n <= 16) ? 16 : ((n <= 64) ? 64 : 256);
            }
            else
            {
                variant    = 2;
                ylocalsize = 64 * ((in_cstride + 63) / 64); //(in_cstride <= 256) ? 256 : 1024;
            }
            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_VARIANT=" + std::to_string(variant);
            parms += " -DMIO_BN_GRP0=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);

            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            xgridsize = c;
            ygridsize = ylocalsize;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << parms << std::endl;
#endif
            bnFwdTrainSelectSingle(handle,
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
        } // end multi / single select
    }
    else
    {

        xlocalsize = 1;
        ylocalsize = 256;
        zlocalsize = 1;
        vld.push_back(xlocalsize);
        vld.push_back(ylocalsize);
        vld.push_back(zlocalsize);

        parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);

        auto segment = std::ceil(double(in_cstride) / double(ylocalsize));

        xgridsize = c;
        ygridsize = segment * ylocalsize;
        zgridsize = 1;
        vgd.push_back(xgridsize);
        vgd.push_back(ygridsize);
        vgd.push_back(zgridsize);

        program_name += "PerAct.cl";
        kernel_name += "PerActivation";

#if(MIOPEN_BN_CPP_DEBUG == 1)
        std::cout << kernel_name << ":: ";
        std::cout << parms << std::endl;
#endif
        if(resultsave && resultrunning)
        {
            handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
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
            handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x,
                in_nstride,
                in_cstride,
                y,
                bnScale,
                bnBias,
                expAvgFactor,
                epsilon,
                resultSaveMean,
                resultSaveInvVariance);
        }
        else if(resultrunning)
        {
            handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
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
            handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
                x, in_nstride, in_cstride, y, bnScale, bnBias, expAvgFactor, epsilon);
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

        std::string algo_name    = "miopenBatchNormalizationForwardInference";
        std::string program_name = "MIOpenBatchNormFwdInfer"; // build this up
        std::string kernel_name  = "BatchNormFwdInfer";
        std::string network_config{};
        std::string parms{}; // compiler parameters

        int n, c, h, w;
        std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

        unsigned int in_nstride = c * h * w;
        unsigned int in_cstride = h * w;

        parms += "-DMIO_BN_N=" + std::to_string(n);
        parms += " -DMIO_BN_HW=" + std::to_string(in_cstride);
        parms += " -DMIO_BN_CHW=" + std::to_string(in_nstride);

        size_t xlocalsize = 0;
        size_t ylocalsize = 0;
        size_t zlocalsize = 0;

        size_t xgridsize = 0;
        size_t ygridsize = 0;
        size_t zgridsize = 0;

        std::vector<size_t> vld;
        std::vector<size_t> vgd;

        if(bn_mode == miopenBNSpatial)
        { // SPATIAL kernels
            program_name += "Spatial.cl";
            kernel_name += "Spatial";
        }
        else
        {
            // PER ACTIVATION
            program_name += "PerAct.cl";
            kernel_name += "PerActivation";
        }
        xlocalsize = 1;
        ylocalsize = (in_cstride > 1024) ? 1024 : ((64 >= in_cstride) ? 64 : 256);
        zlocalsize = 1;
        vld.push_back(xlocalsize);
        vld.push_back(ylocalsize);
        vld.push_back(zlocalsize);

        auto segment = std::ceil(double(in_cstride) / double(ylocalsize));

        xgridsize = c;
        ygridsize = segment * ylocalsize;
        zgridsize = 1;
        vgd.push_back(xgridsize);
        vgd.push_back(ygridsize);
        vgd.push_back(zgridsize);
        kernel_name += "Est";

        parms += " -DMIO_BN_GRP0=" + std::to_string(1);
        parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
        parms += " -DMIO_BN_GRP2=" + std::to_string(1);

#if(MIOPEN_BN_CPP_DEBUG == 1)
        std::cout << kernel_name << ":: ";
        std::cout << parms << std::endl;
#endif
        handle.GetKernel(algo_name, network_config, program_name, kernel_name, vld, vgd, parms)(
            x, y, estimatedMean, estimatedVariance, bnScale, bnBias, epsilon);
    }
    else
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

    //#if(MIO_BN_TIME_EVERYTHING == 1)
    auto t_start = std::chrono::high_resolution_clock::now();
    //#endif
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

    std::string algo_name    = "miopenBatchNormalizationBackwardProp";
    std::string program_name = "MIOpenBatchNormBwd"; // build this up
    std::string kernel_name  = "BatchNormBwd";
    std::string network_config{};
    std::string parms{};

    int n, c, h, w;
    std::tie(n, c, h, w) = tien<4>(xDesc.GetLengths());

    unsigned int in_cstride = h * w;
    unsigned int in_nstride = c * in_cstride;
    unsigned int in_nhw     = n * in_cstride;
    unsigned int in_nchw    = n * in_nstride;

    auto inhw = float(1.0 / in_nhw);

    parms += "-DMIO_BN_N=" + std::to_string(n);
    parms += " -DMIO_BN_C=" + std::to_string(c);
    parms += " -DMIO_BN_HW=" + std::to_string(in_cstride);
    parms += " -DMIO_BN_NCHW=" + std::to_string(in_nchw);
    parms += " -DMIO_BN_NHW=" + std::to_string(in_nhw);
    parms += " -DMIO_BN_CHW=" + std::to_string(in_nstride);

    size_t xlocalsize;
    size_t ylocalsize;
    size_t zlocalsize;

    size_t xgridsize;
    size_t ygridsize;
    size_t zgridsize;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    bool useSaved = false;

    if(bn_mode == miopenBNSpatial)
    { // SPATIAL kernels

        if(savedMean != nullptr && savedInvVariance != nullptr)
        {
            useSaved = true;
            parms += " -DMIO_BN_USESAVED=1";
        }
        else
        {
            useSaved = false;
            parms += " -DMIO_BN_USESAVED=0";
        }

        program_name += "Spatial.cl";
        kernel_name += "Spatial";

        if(in_cstride > 1024 && in_nhw < 33554432)
        {
            // unsigned int variant = (in_cstride < 2097152)? 5: 6;
            unsigned int variant = (h == w) ? 5 : 6;

            xlocalsize = 1024;
            ylocalsize = 1;
            zlocalsize = 1;

            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_VARIANT=" + std::to_string(variant);
            parms += " -DMIO_BN_GRP0=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);

            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            xgridsize = xlocalsize * c;
            ygridsize = 1;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << kernel_name << ":: ";
            std::cout << parms << std::endl;
#endif
            bnBwdTrainSelectSingle(handle,
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
        else if(in_cstride <= 512 && n > 3 && in_cstride > 4)
        {
            unsigned int variant = 0;
            xlocalsize           = 1024;
            ylocalsize           = 1;
            zlocalsize           = 1;

            unsigned int segment = in_cstride * (xlocalsize / in_cstride);
            unsigned int nloops  = (in_nhw + segment - 1) / segment;

            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_VARIANT=" + std::to_string(variant);
            parms += " -DMIO_BN_GRP0=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);
            parms += " -DMIO_BN_NLOOP=" + std::to_string(nloops);
            parms += " -DMIO_BN_SEGMENT=" + std::to_string((segment > in_nhw) ? in_nhw : segment);

            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            xgridsize = 1024 * c;
            ygridsize = 1;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << kernel_name << ":: ";
            std::cout << parms << std::endl;
#endif
            bnBwdTrainSelectSingle(handle,
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
        else if(in_cstride > 1024)
        {
            unsigned int variant = 4;
            xlocalsize           = 1;
            ylocalsize           = 1024;
            zlocalsize           = 1;
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            auto segment = int(std::ceil(double(in_cstride) / double(ylocalsize)));
            xgridsize    = c;
            ygridsize    = segment * ylocalsize;
            zgridsize    = 1;

            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            unsigned int numwgs = std::ceil(float(ygridsize) / ylocalsize);
            parms += " -DMIO_BN_NGRPS=" + std::to_string(numwgs);
            parms += " -DMIO_BN_VARIANT=" + std::to_string(variant);
            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP0=" + std::to_string(1);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(1);
            parms += " -DMIO_BN_SEGMENT=" + std::to_string(segment);

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << kernel_name << ":: ";
            std::cout << parms << std::endl;
#endif
            parms += " -DMIO_BN_VARIANT=" + std::to_string(variant);
            // MULTI

            //#if(MIO_BN_TIME_EVERYTHING == 1)
            auto t_end = std::chrono::high_resolution_clock::now();

            std::cout << "Wall clock: PREAMBLE: "
                      << std::chrono::duration<double>(t_end - t_start).count() * 1000.0 << " ms."
                      << std::endl;
            //#endif
            bnBwdTrainSelectMulti(handle,
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
        else
        {
            xlocalsize = 1;
            zlocalsize = 1;

            unsigned int variant;

            if(in_cstride < 257 && in_cstride > n && n <= 64 && in_cstride > 1)
            {
                variant    = 1;
                ylocalsize = (in_cstride <= 16) ? 16 : ((in_cstride <= 64) ? 64 : 256);
            }
            else if(in_cstride <= 64)
            {
                variant    = 2;
                ylocalsize = (n <= 16) ? 16 : ((n <= 64) ? 64 : 256);
            }
            else
            {
                variant    = 3;
                ylocalsize = 64 * ((in_cstride + 63) / 64); //(in_cstride <= 256) ? 256 : 1024;
            }
            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_VARIANT=" + std::to_string(variant);
            parms += " -DMIO_BN_GRP0=" + std::to_string(xlocalsize);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(zlocalsize);

            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            xgridsize = c;
            ygridsize = ylocalsize;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << kernel_name << ":: ";
            std::cout << parms << std::endl;
#endif
            bnBwdTrainSelectSingle(handle,
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

    } // END spatial
    else
    { // PER ACT
        program_name += "PerAct.cl";
        kernel_name += "PerActivation";

        parms += "-DMIO_BN_N=" + std::to_string(n);
        parms += " -DMIO_BN_HW=" + std::to_string(in_cstride);
        parms += " -DMIO_BN_NHW=" + std::to_string(n * h * w);

        xlocalsize = 1;
        ylocalsize = (64 >= in_cstride) ? 64 : 256;
        zlocalsize = 1;
        vld.push_back(xlocalsize);
        vld.push_back(ylocalsize);
        vld.push_back(zlocalsize);

        parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);

        unsigned int segment = std::ceil(double(in_cstride) / double(ylocalsize));

        xgridsize = c;
        ygridsize = segment * ylocalsize;
        zgridsize = 1;
        vgd.push_back(xgridsize);
        vgd.push_back(ygridsize);
        vgd.push_back(zgridsize);

        if(savedMean != nullptr && savedInvVariance != nullptr)
        {
            useSaved = true;
            parms += " -DMIO_BN_USESAVED=1";
        }
        else
        {
            useSaved = false;
            parms += " -DMIO_BN_USESAVED=0";
        }

        if(useSaved)
        {
            kernel_name += "Saved";
            handle.GetKernel("miopenBatchNormalizationBwd",
                             network_config,
                             program_name,
                             kernel_name,
                             vld,
                             vgd,
                             parms)(x,
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
            handle.GetKernel("miopenBatchNormalizationBwd",
                             network_config,
                             program_name,
                             kernel_name,
                             vld,
                             vgd,
                             parms)(x,
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
    if(miopen::CheckNumericsEnabled())
    {
        miopen::checkNumericsOutput(handle, dxDesc, dx);
        miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnScaleDiff);
        miopen::checkNumericsOutput(handle, bnScaleBiasDiffDesc, resultBnBiasDiff);
    }
}
} // namespace miopen
