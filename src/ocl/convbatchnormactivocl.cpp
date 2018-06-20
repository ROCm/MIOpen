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
#include <miopen/conv_batch_norm_activ.hpp>
#include <miopen/solver.hpp>
#include <miopen/util.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/visit_float.hpp>
#include <miopen/logger.hpp>
#include "miopen/solver.hpp"
#include <chrono>
#include <miopen/fusion.hpp>

namespace miopen {

void DirectConvInference(Handle& handle,
                         const void* alpha,
                         const TensorDescriptor& xDesc,
                         ConstData_t x,
                         const TensorDescriptor& wDesc,
                         ConstData_t w,
                         const void* beta,
                         const TensorDescriptor& yDesc,
                         Data_t y,
                         int pad_h,
                         int pad_w,
                         int u,
                         int v,
                         int dilation_h,
                         int dilation_w,
                         int bias_mode,
                         ConstData_t convBias)
{
    printf("HERE! : %s", __func__);
    if(x == nullptr || w == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != wDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != wDesc.GetType())
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
        MIOPEN_THROW(miopenStatusNotImplemented, "Only alpha=1 and beta=0 is supported");
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, wDesc, w);
    }

    if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1])
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    // TODO(paul): Replicating code for now.
    mlo_construct_direct2D_fusion construct_params(1); // forward

    construct_params.setOutputDescFromMLDesc(yDesc);
    construct_params.setInputDescFromMLDesc(xDesc);
    construct_params.setWeightDescFromMLDesc(wDesc);
    construct_params.setConvDescr(pad_h, pad_w, u, v, dilation_h, dilation_w);
    construct_params.setStream(&handle);

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);
    network_config += std::to_string(bias_mode);

    std::string algorithm_name = "miopenConvolutionAlgoDirectUni";

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);

    KernelInvoke kernel;

    if(!kernels.empty())
    {
        kernel = kernels.front();
    }
    else
    {
#if 0
        mloConstruct(construct_params);
        std::string program_name       = construct_params.getKernelFile();
        std::string kernel_name        = construct_params.getKernelName();
        const std::string& parms       = construct_params.getCompilerOptions();
        const std::vector<size_t>& vld = construct_params.getLocalWkSize();
        const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();
#else
        ConvolutionContext params;
        construct_params.mloCopyTo(params);
        params.general_compile_options += " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";
        params.general_compile_options += " -DMLO_CONV_BIAS=" + std::to_string(bias_mode);

        auto kernel_info         = solver::CBAFusionGetSolution(params);
        std::string program_name = "MIOpenConvDirUni.cl";
        // std::string program_name       = kernel_info.kernel_file;
        std::string kernel_name = "MIOpenConvUni";
        // std::string kernel_name        = kernel_info.kernel_name;
        const std::string parms        = kernel_info.comp_options;
        const std::vector<size_t>& vld = kernel_info.l_wk;
        const std::vector<size_t>& vgd = kernel_info.g_wk;
#endif

        kernel = handle.AddKernel(
            algorithm_name, network_config, program_name, kernel_name, vld, vgd, parms);
    }

    float padding_val = 0;

    visit_float(xDesc.GetType(), [&](auto as_float) {
        {
            if(bias_mode != 0)
                kernel(x, w, convBias, y, as_float(padding_val));
            else
                kernel(x, w, y, as_float(padding_val));
        }
    });

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
}

void DirectConvBNActivInference(Handle& handle,
                                const void* alpha,
                                const TensorDescriptor& xDesc,
                                ConstData_t x,
                                const TensorDescriptor& wDesc,
                                ConstData_t w,
                                const void* beta,
                                const TensorDescriptor& yDesc,
                                Data_t y,
                                int pad_h,
                                int pad_w,
                                int u,
                                int v,
                                int dilation_h,
                                int dilation_w,
                                int bias_mode,
                                ConstData_t convBias,
                                miopenBatchNormMode_t bn_mode,
                                ConstData_t bnScale,
                                ConstData_t bnBias,
                                ConstData_t estimatedMean,
                                ConstData_t estimatedVariance,
                                double epsilon,
                                miopenActivationMode_t activ_mode,
                                double activ_alpha,
                                double activ_beta,
                                double activ_gama)
{
    printf("HERE! : %s", __func__);
    if(x == nullptr || w == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != wDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != wDesc.GetType())
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
        MIOPEN_THROW(miopenStatusNotImplemented, "Only alpha=1 and beta=0 is supported");
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, wDesc, w);
    }

    if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1])
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    // TODO(paul): Replicating code for now.
    mlo_construct_direct2D_fusion construct_params(1); // forward

    construct_params.setOutputDescFromMLDesc(yDesc);
    construct_params.setInputDescFromMLDesc(xDesc);
    construct_params.setWeightDescFromMLDesc(wDesc);
    construct_params.setConvDescr(pad_h, pad_w, u, v, dilation_h, dilation_w);
    construct_params.setStream(&handle);

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);
    network_config +=
        std::to_string(activ_mode) + std::to_string(bias_mode) + std::to_string(bn_mode);

    std::string algorithm_name = "miopenDirConvBatchNormActivAlgo";

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);

    KernelInvoke kernel;

    if(!kernels.empty())
    {
        kernel = kernels.front();
    }
    else
    {
        ConvolutionContext params;
        construct_params.mloCopyTo(params);
        params.general_compile_options += " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";
        params.general_compile_options += " -DMIOPEN_NRN_OP_ID=" + std::to_string(activ_mode);
        params.general_compile_options += " -DMLO_CONV_BIAS=" + std::to_string(bias_mode);

        if(bn_mode == miopenBNSpatial)
            params.general_compile_options += " -DSPATIAL_BN";
        else if(bn_mode == miopenBNPerActivation)
            params.general_compile_options += " -DPERACT_BN";
        else
            params.general_compile_options += " -DNO_BN";

        auto kernel_info = solver::CBAFusionGetSolution(params);

        std::string program_name       = kernel_info.kernel_file;
        std::string kernel_name        = kernel_info.kernel_name;
        const std::string parms        = kernel_info.comp_options;
        const std::vector<size_t>& vld = kernel_info.l_wk;
        const std::vector<size_t>& vgd = kernel_info.g_wk;

        kernel = handle.AddKernel(
            algorithm_name, network_config, program_name, kernel_name, vld, vgd, parms);
    }

    float padding_val = 0;

    visit_float(xDesc.GetType(), [&](auto as_float) {
        auto f_activ_alpha = as_float(activ_alpha);
        auto f_activ_beta  = as_float(activ_beta);
        auto f_activ_gama  = as_float(activ_gama);

        if(bias_mode != 0)
        {
            kernel(x,
                   w,
                   convBias,
                   y,
                   as_float(padding_val),
                   estimatedMean,
                   estimatedVariance,
                   bnScale,
                   bnBias,
                   epsilon,
                   f_activ_gama,
                   f_activ_alpha,
                   f_activ_beta);
        }
        else
        {
            kernel(x,
                   w,
                   y,
                   as_float(padding_val),
                   estimatedMean,
                   estimatedVariance,
                   bnScale,
                   bnBias,
                   epsilon,
                   f_activ_gama,
                   f_activ_alpha,
                   f_activ_beta);
        }
    });

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
}

void DirectConvActivInference(Handle& handle,
                              const void* alpha,
                              const TensorDescriptor& xDesc,
                              ConstData_t x,
                              const TensorDescriptor& wDesc,
                              ConstData_t w,
                              const void* beta,
                              const TensorDescriptor& yDesc,
                              Data_t y,
                              int pad_h,
                              int pad_w,
                              int u,
                              int v,
                              int dilation_h,
                              int dilation_w,
                              int bias_mode,
                              ConstData_t convBias,
                              miopenActivationMode_t activ_mode,
                              double activ_alpha,
                              double activ_beta,
                              double activ_gama)
{

    printf("HERE! : %s", __func__);
    if(x == nullptr || w == nullptr || y == nullptr)
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetSize() != yDesc.GetSize() || xDesc.GetSize() != wDesc.GetSize())
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }
    if(xDesc.GetType() != yDesc.GetType() || xDesc.GetType() != wDesc.GetType())
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
        MIOPEN_THROW(miopenStatusNotImplemented, "Only alpha=1 and beta=0 is supported");
    }

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsInput(handle, xDesc, x);
        miopen::checkNumericsInput(handle, wDesc, w);
    }

    if(xDesc.GetLengths()[1] != wDesc.GetLengths()[1])
    {
        MIOPEN_THROW(miopenStatusBadParm);
    }

    // TODO(paul): Replicating code for now.
    mlo_construct_direct2D_fusion construct_params(1); // forward

    construct_params.setOutputDescFromMLDesc(yDesc);
    construct_params.setInputDescFromMLDesc(xDesc);
    construct_params.setWeightDescFromMLDesc(wDesc);
    construct_params.setConvDescr(pad_h, pad_w, u, v, dilation_h, dilation_w);
    construct_params.setStream(&handle);

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);
    network_config += std::to_string(activ_mode) + std::to_string(bias_mode);

    std::string algorithm_name = "miopenDirConvBatchNormActivAlgo";

    auto&& kernels = handle.GetKernels(algorithm_name, network_config);

    KernelInvoke kernel;

    if(!kernels.empty())
    {
        kernel = kernels.front();
    }
    else
    {
        ConvolutionContext params;
        construct_params.mloCopyTo(params);
        params.general_compile_options += " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";
        params.general_compile_options += " -DMIOPEN_NRN_OP_ID=" + std::to_string(activ_mode);
        params.general_compile_options += " -DMLO_CONV_BIAS=" + std::to_string(bias_mode);
        params.general_compile_options += " -DNO_BN";

        auto kernel_info = solver::CBAFusionGetSolution(params);

        std::string program_name       = kernel_info.kernel_file;
        std::string kernel_name        = kernel_info.kernel_name;
        const std::string parms        = kernel_info.comp_options;
        const std::vector<size_t>& vld = kernel_info.l_wk;
        const std::vector<size_t>& vgd = kernel_info.g_wk;

        kernel = handle.AddKernel(
            algorithm_name, network_config, program_name, kernel_name, vld, vgd, parms);
    }

    float padding_val = 0;

    visit_float(xDesc.GetType(), [&](auto as_float) {
        auto f_activ_alpha = as_float(activ_alpha);
        auto f_activ_beta  = as_float(activ_beta);
        auto f_activ_gama  = as_float(activ_gama);

        if(bias_mode != 0)
        {
            kernel(x,
                   w,
                   convBias,
                   y,
                   as_float(padding_val),
                   f_activ_gama,
                   f_activ_alpha,
                   f_activ_beta);
        }
        else
        {
            kernel(x, w, y, as_float(padding_val), f_activ_gama, f_activ_alpha, f_activ_beta);
        }
    });

    if(miopen::CheckNumericsEnabled() != 0)
    {
        miopen::checkNumericsOutput(handle, yDesc, y);
    }
}
}
