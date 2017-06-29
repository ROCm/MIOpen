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

namespace miopen {

void BatchNormForwardTraining(Handle& handle,
                              miopenBatchNormMode_t bn_mode,
                              const void* /* alpha */,
                              const void* /* beta  */,
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

    std::string program_name = "MIOpenBatchNormFwdTrain";
    std::string algo_name    = "miopenBatchNormalizationForwardTraining";
    std::string kernel_name  = "BatchNormFwdTrain";
    std::string kernel_subname{};
    std::string network_config{};

    int n, c, h, w;
    std::tie(n, c, h, w) = tie4(xDesc.GetLengths());

    unsigned int in_nstride = c * h * w;
    unsigned int in_cstride = h * w;

    size_t xlocalsize;
    size_t ylocalsize;
    size_t zlocalsize;

    size_t xgridsize;
    size_t ygridsize;
    size_t zgridsize;

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
    parms += " -DMIO_BN_NHW=" + std::to_string(n * h * w);
    parms += " -DMIO_BN_CHW=" + std::to_string(in_nstride);

    float ktime = 0.;
    float ctime = 0.;
    if(handle.IsProfilingEnabled())
    {
        handle.ResetKernelTime();
    }

    unsigned int segment;
    auto inhw = float(1.0 / (n * h * w));

    if(bn_mode == miopenBNSpatial)
    {

        program_name += "Spatial.cl";
        kernel_name += "Spatial";

        if(in_cstride < n && n <= 64)
        {

            xlocalsize = 1;
            ylocalsize = 64;
            zlocalsize = 1;
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            xgridsize = c;
            ygridsize = ylocalsize;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            unsigned int numwgs = std::ceil(float(ygridsize) / ylocalsize);
            parms += " -DMIO_BN_NGRPS=" + std::to_string(numwgs);
#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << " -DMIO_BN_VARIANT=0" << std::endl;
#endif
            parms += " -DMIO_BN_VARIANT=0";
            parms += " -DMIO_BN_LDS_NSIZE=" + std::to_string(n);
            parms += " -DMIO_BN_LDS_HWSIZE=" + std::to_string(in_cstride);
            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP0=" + std::to_string(1);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(1);

            if(resultsave && resultrunning)
            {
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_name,
                                 vld,
                                 vgd,
                                 parms)(x,
                                        y,
                                        bnScale,
                                        bnBias,
                                        inhw,
                                        expAvgFactor,
                                        resultRunningMean,
                                        resultRunningVariance,
                                        epsilon,
                                        resultSaveMean,
                                        resultSaveInvVariance);
            }
            else if(resultsave)
            {
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_name,
                                 vld,
                                 vgd,
                                 parms)(
                    x, y, bnScale, bnBias, inhw, epsilon, resultSaveMean, resultSaveInvVariance);
            }
            else if(resultrunning)
            {
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_name,
                                 vld,
                                 vgd,
                                 parms)(x,
                                        y,
                                        bnScale,
                                        bnBias,
                                        inhw,
                                        expAvgFactor,
                                        resultRunningMean,
                                        resultRunningVariance,
                                        epsilon);
            }
            else
            {
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_name,
                                 vld,
                                 vgd,
                                 parms)(x, y, bnScale, bnBias, inhw, epsilon);
            }
            return;
        }

        if(in_cstride < 64)
        {

            xlocalsize = 1;
            ylocalsize = 64;
            zlocalsize = 1;
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            xgridsize = c;
            ygridsize = ylocalsize;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            unsigned int numwgs = std::ceil(float(ygridsize) / ylocalsize);
            parms += " -DMIO_BN_NGRPS=" + std::to_string(numwgs);
            auto lds_size =
                static_cast<unsigned long long>(((in_cstride * (n + 2) + 1) / 4096) * numwgs);

            if(lds_size <= 1)
            {
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=1" << std::endl;
#endif
                parms += " -DMIO_BN_VARIANT=1";
                parms += " -DMIO_BN_LDS_NSIZE=" + std::to_string(n);
                parms += " -DMIO_BN_LDS_HWSIZE=" + std::to_string(in_cstride);
                parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
                parms += " -DMIO_BN_GRP0=" + std::to_string(1);
                parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
                parms += " -DMIO_BN_GRP2=" + std::to_string(1);

                if(resultsave && resultrunning)
                {
                    handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                     network_config,
                                     program_name,
                                     kernel_name,
                                     vld,
                                     vgd,
                                     parms)(x,
                                            y,
                                            bnScale,
                                            bnBias,
                                            inhw,
                                            expAvgFactor,
                                            resultRunningMean,
                                            resultRunningVariance,
                                            epsilon,
                                            resultSaveMean,
                                            resultSaveInvVariance);
                }
                else if(resultsave)
                {
                    handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                     network_config,
                                     program_name,
                                     kernel_name,
                                     vld,
                                     vgd,
                                     parms)(x,
                                            y,
                                            bnScale,
                                            bnBias,
                                            inhw,
                                            epsilon,
                                            resultSaveMean,
                                            resultSaveInvVariance);
                }
                else if(resultrunning)
                {
                    handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                     network_config,
                                     program_name,
                                     kernel_name,
                                     vld,
                                     vgd,
                                     parms)(x,
                                            y,
                                            bnScale,
                                            bnBias,
                                            inhw,
                                            expAvgFactor,
                                            resultRunningMean,
                                            resultRunningVariance,
                                            epsilon);
                }
                else
                {
                    handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                     network_config,
                                     program_name,
                                     kernel_name,
                                     vld,
                                     vgd,
                                     parms)(x, y, bnScale, bnBias, inhw, epsilon);
                }
            }
            else
            {
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=2" << std::endl;
#endif
                parms += " -DMIO_BN_VARIANT=2";
                parms += " -DMIO_BN_GRP0=" + std::to_string(1);
                parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
                parms += " -DMIO_BN_GRP2=" + std::to_string(1);
                parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);

                if(resultsave && resultrunning)
                {
                    handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                     network_config,
                                     program_name,
                                     kernel_name,
                                     vld,
                                     vgd,
                                     parms)(x,
                                            y,
                                            bnScale,
                                            bnBias,
                                            inhw,
                                            expAvgFactor,
                                            resultRunningMean,
                                            resultRunningVariance,
                                            epsilon,
                                            resultSaveMean,
                                            resultSaveInvVariance);
                }
                else if(resultsave)
                {
                    handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                     network_config,
                                     program_name,
                                     kernel_name,
                                     vld,
                                     vgd,
                                     parms)(x,
                                            y,
                                            bnScale,
                                            bnBias,
                                            inhw,
                                            epsilon,
                                            resultSaveMean,
                                            resultSaveInvVariance);
                }
                else if(resultrunning)
                {
                    handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                     network_config,
                                     program_name,
                                     kernel_name,
                                     vld,
                                     vgd,
                                     parms)(x,
                                            y,
                                            bnScale,
                                            bnBias,
                                            inhw,
                                            expAvgFactor,
                                            resultRunningMean,
                                            resultRunningVariance,
                                            epsilon);
                }
                else
                {
                    handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                     network_config,
                                     program_name,
                                     kernel_name,
                                     vld,
                                     vgd,
                                     parms)(x, y, bnScale, bnBias, inhw, epsilon);
                }
            }
        }
        else if(in_cstride <= 256)
        {

            xlocalsize = 1;
            ylocalsize = 256;
            zlocalsize = 1;
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            xgridsize = c;
            ygridsize = ylocalsize;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            unsigned int numwgs = std::ceil(float(ygridsize) / ylocalsize);
            parms += " -DMIO_BN_NGRPS=" + std::to_string(numwgs);
            auto lds_size =
                static_cast<unsigned long long>(((in_cstride * (n + 2) + 1) / 8192) * numwgs);
            if(lds_size > 1)
            {
                parms += " -DMIO_BN_VARIANT=3";
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=3" << std::endl;
#endif
            }
            else
            {
                parms += " -DMIO_BN_VARIANT=4";
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=4" << std::endl;
#endif
                parms += " -DMIO_BN_LDS_NSIZE=" + std::to_string(n);
                parms += " -DMIO_BN_LDS_HWSIZE=" + std::to_string(h * w);
            }
            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP0=" + std::to_string(1);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(1);

            if(resultsave && resultrunning)
            {
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_name,
                                 vld,
                                 vgd,
                                 parms)(x,
                                        y,
                                        bnScale,
                                        bnBias,
                                        inhw,
                                        expAvgFactor,
                                        resultRunningMean,
                                        resultRunningVariance,
                                        epsilon,
                                        resultSaveMean,
                                        resultSaveInvVariance);
            }
            else if(resultsave)
            {
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_name,
                                 vld,
                                 vgd,
                                 parms)(
                    x, y, bnScale, bnBias, inhw, epsilon, resultSaveMean, resultSaveInvVariance);
            }
            else if(resultrunning)
            {
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_name,
                                 vld,
                                 vgd,
                                 parms)(x,
                                        y,
                                        bnScale,
                                        bnBias,
                                        inhw,
                                        expAvgFactor,
                                        resultRunningMean,
                                        resultRunningVariance,
                                        epsilon);
            }
            else
            {
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_name,
                                 vld,
                                 vgd,
                                 parms)(x, y, bnScale, bnBias, inhw, epsilon);
            }
            return;
        }
        else
        {

            xlocalsize = 1;
            ylocalsize = 256;
            zlocalsize = 1;
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            segment   = std::ceil(double(in_cstride) / double(ylocalsize));
            xgridsize = c;
            ygridsize = segment * ylocalsize;
            zgridsize = 1;

            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            unsigned int numwgs = std::ceil(float(ygridsize) / ylocalsize);
            parms += " -DMIO_BN_NGRPS=" + std::to_string(numwgs);
            parms += " -DMIO_BN_VARIANT=5";
#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << " -DMIO_BN_VARIANT=5" << std::endl;
#endif
            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP0=" + std::to_string(1);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(1);

            if(resultsave && resultrunning)
            {

                // Run mean reduction kernel
                kernel_subname = kernel_name + "Mean";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);

                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime = ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalMean";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(y, inhw, expAvgFactor, resultRunningMean, resultSaveMean);

                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                // Run variance reduction kernel
                kernel_subname = kernel_name + "Variance";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);

                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalVariance";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(
                    y, inhw, expAvgFactor, resultRunningVariance, epsilon, resultSaveInvVariance);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                // Run norm kernel
                kernel_subname = kernel_name + "Norm";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y, bnScale, bnBias);

                if(handle.IsProfilingEnabled())
                {
                    handle.GetKernelTime();
                    handle.AccumKernelTime(ctime);
                }
            }
            else if(resultsave)
            {

                kernel_subname = kernel_name + "Mean";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);

                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime = ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalMean";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(y, inhw, resultSaveMean);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "Variance";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalVariance";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(y, inhw, epsilon, resultSaveInvVariance);

                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "Norm";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y, bnScale, bnBias);
                if(handle.IsProfilingEnabled())
                {
                    handle.GetKernelTime();
                    handle.AccumKernelTime(ctime);
                }
            }
            else if(resultrunning)
            {

                kernel_subname = kernel_name + "Mean";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime = ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalMean";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(y, inhw, expAvgFactor, resultRunningMean);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "Variance";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalVariance";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(y, inhw, expAvgFactor, resultRunningVariance, epsilon);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "Norm";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y, bnScale, bnBias);
                if(handle.IsProfilingEnabled())
                {
                    handle.GetKernelTime();
                    handle.AccumKernelTime(ctime);
                }
            }
            else
            {

                kernel_subname = kernel_name + "Mean";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime = ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalMean";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(y, inhw);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "Variance";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalVariance";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(y, inhw, epsilon);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "Norm";
                handle.GetKernel("miopenBatchNormalizationForwardTraining",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y, bnScale, bnBias);
                if(handle.IsProfilingEnabled())
                {
                    handle.GetKernelTime();
                    handle.AccumKernelTime(ctime);
                }
            }
        }
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

        segment = std::ceil(double(in_cstride) / double(ylocalsize));

        xgridsize = c;
        ygridsize = segment * ylocalsize;
        zgridsize = 1;
        vgd.push_back(xgridsize);
        vgd.push_back(ygridsize);
        vgd.push_back(zgridsize);

        program_name += "PerAct.cl";
        kernel_name += "PerActivation";
        if(resultsave && resultrunning)
        {
            handle.GetKernel("miopenBatchNormalizationForwardTraining",
                             network_config,
                             program_name,
                             kernel_name,
                             vld,
                             vgd,
                             parms)(x,
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
            handle.GetKernel("miopenBatchNormalizationForwardTraining",
                             network_config,
                             program_name,
                             kernel_name,
                             vld,
                             vgd,
                             parms)(x,
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
            handle.GetKernel("miopenBatchNormalizationForwardTraining",
                             network_config,
                             program_name,
                             kernel_name,
                             vld,
                             vgd,
                             parms)(x,
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
            handle.GetKernel("miopenBatchNormalizationForwardTraining",
                             network_config,
                             program_name,
                             kernel_name,
                             vld,
                             vgd,
                             parms)(
                x, in_nstride, in_cstride, y, bnScale, bnBias, expAvgFactor, epsilon);
        }
    } // end per-activation
}
//================== END FWD TRAIN ===================

//============ BEGIN FORWARD INFERENCE ===============
void BatchNormForwardInference(Handle& handle,
                               miopenBatchNormMode_t bn_mode,
                               const void* /* alpha */,
                               const void* /* beta */,
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

    std::string program_name = "MIOpenBatchNormFwdInfer"; // build this up
    std::string kernel_name  = "BatchNormFwdInfer";
    std::string kernel_subname{};
    std::string network_config{};

    int n, c, h, w;
    std::tie(n, c, h, w) = tie4(xDesc.GetLengths());

    unsigned int in_nstride = c * h * w;
    unsigned int in_cstride = h * w;

    size_t xlocalsize;
    size_t ylocalsize;
    size_t zlocalsize;

    size_t xgridsize;
    size_t ygridsize;
    size_t zgridsize;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    float ktime = 0.;
    float ctime = 0.;
    if(handle.IsProfilingEnabled())
    {
        handle.ResetKernelTime();
    }

    // compile parameters
    std::string parms{};
    bool useEstimated = false;
    if(estimatedMean != nullptr && estimatedVariance != nullptr)
    {
        useEstimated = true;
    }

    if(bn_mode == miopenBNSpatial)
    { // SPATIAL kernels
        program_name += "Spatial.cl";
        kernel_name += "Spatial";
        parms += "-DMIO_BN_N=" + std::to_string(n);
        parms += " -DMIO_BN_C=" + std::to_string(c);
        parms += " -DMIO_BN_HW=" + std::to_string(in_cstride);
        parms += " -DMIO_BN_NHW=" + std::to_string(n * h * w);
        parms += " -DMIO_BN_CHW=" + std::to_string(in_nstride);

        unsigned int segment;
        unsigned int numwgs;
        auto inhw = double(1.0 / (n * h * w));

        if(useEstimated)
        {

            xlocalsize = 1;
            ylocalsize = (in_cstride > 1024) ? ((64 >= in_cstride) ? 64 : 256) : 1024;
            zlocalsize = 1;
            vld.push_back(xlocalsize);
            vld.push_back(ylocalsize);
            vld.push_back(zlocalsize);

            parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP0=" + std::to_string(1);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(1);

            segment = std::ceil(double(in_cstride) / double(ylocalsize));

            xgridsize = c;
            ygridsize = segment * ylocalsize;
            zgridsize = 1;
            vgd.push_back(xgridsize);
            vgd.push_back(ygridsize);
            vgd.push_back(zgridsize);

            numwgs = std::ceil(float(ygridsize) / ylocalsize);
            parms += " -DMIO_BN_NGRPS=" + std::to_string(numwgs);

            kernel_name += "Est";
            parms += " -DMIO_BN_VARIANT=0";
#if(MIOPEN_BN_CPP_DEBUG == 1)
            std::cout << " -DMIO_BN_VARIANT=0" << std::endl;
#endif
            handle.GetKernel("miopenBatchNormalizationForwardInference",
                             network_config,
                             program_name,
                             kernel_name,
                             vld,
                             vgd,
                             parms)(
                x, y, estimatedMean, estimatedVariance, bnScale, bnBias, epsilon);
        }
        else
        {

            if(in_cstride <= 256)
            {

                xlocalsize = 1;
                ylocalsize = (64 >= in_cstride) ? 64 : 256;
                zlocalsize = 1;
                vld.push_back(xlocalsize);
                vld.push_back(ylocalsize);
                vld.push_back(zlocalsize);

                parms += " -DMIO_BN_LDS_SIZE=" + std::to_string(ylocalsize);
                parms += " -DMIO_BN_GRP0=" + std::to_string(1);
                parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
                parms += " -DMIO_BN_GRP2=" + std::to_string(1);

                xgridsize = c;
                ygridsize = ylocalsize;
                zgridsize = 1;
                vgd.push_back(xgridsize);
                vgd.push_back(ygridsize);
                vgd.push_back(zgridsize);

                kernel_subname = kernel_name + "SingleNorm";
                parms += " -DMIO_BN_VARIANT=1";
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=1" << std::endl;
#endif
                handle.GetKernel("miopenBatchNormalizationForwardInference",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y, bnScale, bnBias, epsilon, inhw);
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
                parms += " -DMIO_BN_GRP0=" + std::to_string(1);
                parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
                parms += " -DMIO_BN_GRP2=" + std::to_string(1);

                segment = std::ceil(double(in_cstride) / double(ylocalsize));

                xgridsize = c;
                ygridsize = segment * ylocalsize;
                zgridsize = 1;
                vgd.push_back(xgridsize);
                vgd.push_back(ygridsize);
                vgd.push_back(zgridsize);

                numwgs = std::ceil(float(ygridsize) / ylocalsize);
                parms += " -DMIO_BN_NGRPS=" + std::to_string(numwgs);

                parms += " -DMIO_BN_VARIANT=2";
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=2" << std::endl;
#endif
                kernel_subname = kernel_name + "Mean";
                handle.GetKernel("miopenBatchNormalizationForwardInference",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime = ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalMean";
                handle.GetKernel("miopenBatchNormalizationForwardInference",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(y);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "Variance";
                handle.GetKernel("miopenBatchNormalizationForwardInference",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalVariance";
                handle.GetKernel("miopenBatchNormalizationForwardInference",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(y, epsilon);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "Norm";
                handle.GetKernel("miopenBatchNormalizationForwardInference",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, y, bnScale, bnBias);
                if(handle.IsProfilingEnabled())
                {
                    handle.GetKernelTime();
                    handle.AccumKernelTime(ctime);
                }
            }
        }
        // end spatial
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

        unsigned int segment = std::ceil(double(in_cstride) / double(ylocalsize));

        xgridsize = c;
        ygridsize = segment * ylocalsize;
        zgridsize = 1;
        vgd.push_back(xgridsize);
        vgd.push_back(ygridsize);
        vgd.push_back(zgridsize);

        program_name += "PerAct.cl";
        kernel_name += "PerActivation";
        if(useEstimated)
        {
            kernel_name += "Est";
            handle.GetKernel("miopenBatchNormalizationForwardInference",
                             network_config,
                             program_name,
                             kernel_name,
                             vld,
                             vgd,
                             parms)(x,
                                    n,
                                    in_nstride,
                                    in_cstride,
                                    y,
                                    bnScale,
                                    bnBias,
                                    estimatedMean,
                                    estimatedVariance,
                                    epsilon);
        }
        else
        {
            handle.GetKernel("miopenBatchNormalizationForwardInference",
                             network_config,
                             program_name,
                             kernel_name,
                             vld,
                             vgd,
                             parms)(x, n, in_nstride, in_cstride, y, bnScale, bnBias, epsilon);
        }
    } // end per-activation
}
//================= END FORWARD INFERENCE ====================

//=============== BEGIN BACKWARDS PROPAGATION ================
void BatchNormBackward(Handle& handle,
                       miopenBatchNormMode_t bn_mode,
                       const void* /* alphaDataDiff */,
                       const void* /* betaDataDiff */,
                       const void* /* alphaParamDiff */,
                       const void* /* betaParamDiff */,
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

    std::string program_name = "MIOpenBatchNormBwd"; // build this up
    std::string kernel_name  = "BatchNormBwd";
    std::string kernel_subname{};
    std::string network_config{};

    int n, c, h, w;
    std::tie(n, c, h, w) = tie4(xDesc.GetLengths());

    unsigned int in_nstride = c * h * w;
    unsigned int in_cstride = h * w;

    size_t xlocalsize;
    size_t ylocalsize;
    size_t zlocalsize;

    size_t xgridsize;
    size_t ygridsize;
    size_t zgridsize;

    std::vector<size_t> vld;
    std::vector<size_t> vgd;

    // compile parameters
    std::string parms = " ";
    bool useSaved     = false;
    if(savedMean != nullptr && savedInvVariance != nullptr)
    {
        useSaved = true;
    }

    float ktime = 0.;
    float ctime = 0.;
    if(handle.IsProfilingEnabled())
    {
        handle.ResetKernelTime();
    }

    if(bn_mode == miopenBNSpatial)
    { // SPATIAL kernels
        program_name += "Spatial.cl";
        kernel_name += "Spatial";
        parms += "-DMIO_BN_N=" + std::to_string(n);
        parms += " -DMIO_BN_C=" + std::to_string(c);
        parms += " -DMIO_BN_HW=" + std::to_string(in_cstride);
        parms += " -DMIO_BN_NHW=" + std::to_string(n * h * w);
        parms += " -DMIO_BN_CHW=" + std::to_string(in_nstride);

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

        auto inhw           = float(1.0 / (n * h * w));
        unsigned int numwgs = std::ceil(float(ygridsize) / ylocalsize);
        parms += " -DMIO_BN_NGRPS=" + std::to_string(numwgs);

        if(useSaved)
        {
            kernel_name += "Saved";
            parms += " -DMIO_BN_GRP0=" + std::to_string(1);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(1);

            auto lds_size = static_cast<unsigned long long>(
                ((in_cstride * n + 2 * ylocalsize + 3) / 8192) * numwgs);
            // printf("lds size: %llu\n", 4*lds_size);fflush(nullptr);

            if(lds_size <= 1 && in_cstride <= 256)
            {

#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=2" << std::endl;
#endif
                parms += " -DMIO_BN_VARIANT=2";
                parms += " -DMIO_BN_LDS_NSIZE=" + std::to_string(n);
                parms += " -DMIO_BN_LDS_HWSIZE=" + std::to_string(in_cstride);

                kernel_subname = kernel_name + "SingleLDSDX";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x,
                                        dy,
                                        dx,
                                        bnScale,
                                        resultBnScaleDiff,
                                        resultBnBiasDiff,
                                        savedMean,
                                        savedInvVariance,
                                        inhw);
            }
            else if(in_cstride < 32 && n <= ylocalsize)
            {
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=0" << std::endl;
#endif
                parms += " -DMIO_BN_VARIANT=0";
                parms += " -DMIO_BN_LDS_NSIZE=" + std::to_string(n);
                parms += " -DMIO_BN_LDS_HWSIZE=" + std::to_string(in_cstride);

                kernel_subname = kernel_name + "SingleDX";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x,
                                        dy,
                                        dx,
                                        bnScale,
                                        resultBnScaleDiff,
                                        resultBnBiasDiff,
                                        savedMean,
                                        savedInvVariance,
                                        inhw);
            }
            else if(in_cstride <= 256)
            {
                parms += " -DMIO_BN_VARIANT=4";
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=4" << std::endl;
#endif
                kernel_subname = kernel_name + "SingleDX";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x,
                                        dy,
                                        dx,
                                        bnScale,
                                        resultBnScaleDiff,
                                        resultBnBiasDiff,
                                        savedMean,
                                        savedInvVariance,
                                        inhw);
            }
            else
            {
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=6" << std::endl;
#endif
                parms += " -DMIO_BN_VARIANT=6";
                kernel_subname = kernel_name + "DBias";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(dy, dx);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime = ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "DScale";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, dy, savedMean, savedInvVariance, dx);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalDBias";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(dx, resultBnBiasDiff);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalDScale";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(dx, resultBnScaleDiff);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "DX";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x,
                                        dy,
                                        dx,
                                        bnScale,
                                        resultBnScaleDiff,
                                        resultBnBiasDiff,
                                        savedMean,
                                        savedInvVariance);
                if(handle.IsProfilingEnabled())
                {
                    handle.GetKernelTime();
                    handle.AccumKernelTime(ctime);
                }
            }
        }
        else
        {

            parms += " -DMIO_BN_GRP0=" + std::to_string(1);
            parms += " -DMIO_BN_GRP1=" + std::to_string(ylocalsize);
            parms += " -DMIO_BN_GRP2=" + std::to_string(1);
            auto lds_size = static_cast<unsigned long long>(
                ((in_cstride * n + 2 * ylocalsize + 3) / 8192) * numwgs);
            // printf("lds size: %llu\n", 4* lds_size);fflush(nullptr);

            if(lds_size <= 1 && in_cstride <= 256)
            {
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=1" << std::endl;
#endif
                parms += " -DMIO_BN_VARIANT=1";
                parms += " -DMIO_BN_LDS_NSIZE=" + std::to_string(n);
                parms += " -DMIO_BN_LDS_HWSIZE=" + std::to_string(in_cstride);

                kernel_subname = kernel_name + "SingleLDSDX";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(
                    x, dy, dx, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, inhw);
            }
            else if(in_cstride <= 256)
            {
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=3" << std::endl;
#endif
                parms += " -DMIO_BN_VARIANT=3";
                parms += " -DMIO_BN_LDS_NSIZE=" + std::to_string(n);
                parms += " -DMIO_BN_LDS_HWSIZE=" + std::to_string(in_cstride);
                kernel_subname = kernel_name + "SingleDX";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(
                    x, dy, dx, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, inhw);
            }
            else
            {
#if(MIOPEN_BN_CPP_DEBUG == 1)
                std::cout << " -DMIO_BN_VARIANT=5" << std::endl;
#endif
                parms += " -DMIO_BN_VARIANT=5";
                kernel_subname = kernel_name + "Mean";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, dx);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime = ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }

                kernel_subname = kernel_name + "DBias";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(dy, dx);

                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalDBias";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(dx, resultBnBiasDiff);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalMean";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(dx);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "Variance";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, dx);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalVariance";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(dx, epsilon);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "DScale";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, dy, dx);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "FinalDScale";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(dx, resultBnScaleDiff);
                if(handle.IsProfilingEnabled())
                {
                    ktime = handle.GetKernelTime();
                    ctime += ktime;
#if(MIO_BN_CPP_PROF == 1)
                    printf("ktime: %f\n", ktime);
                    printf("ctime: %f\n", ctime);
#endif
                }
                else
                {
                    handle.Finish();
                }

                kernel_subname = kernel_name + "DX";
                handle.GetKernel("miopenBatchNormalizationBwd",
                                 network_config,
                                 program_name,
                                 kernel_subname,
                                 vld,
                                 vgd,
                                 parms)(x, dy, dx, bnScale, resultBnScaleDiff, resultBnBiasDiff);
                if(handle.IsProfilingEnabled())
                {
                    handle.GetKernelTime();
                    handle.AccumKernelTime(ctime);
                }
            }
        }
    }
    else
    {
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
}
} // namespace miopen
