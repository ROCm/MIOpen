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

namespace miopen {

namespace solver {

KernelInfo GetSolution(const ConvolutionContext& params)
{
    ConvOclDirectFwdLegacyExhaustiveSearch ConvOclDirectSearch;
    LegacyPerformanceConfig searched_params = ConvOclDirectSearch.GetPerformanceConfig(params);

    ConvSolution result;

    // std::size_t localMemSize = params.stream.GetLocalMemorySize();

    searched_params.CopyTo(result);
    auto pad0 = params.pad0;
    auto pad1 = params.pad1;

    auto hw_wave_sz = 64;
    // auto dev_local_mem_sz = localMemSize; // in bytes

    if(!params.direction.IsForward())
    {
        // backward
        pad0 = params.kernel_size0 - 1 - pad0;
        pad1 = params.kernel_size1 - 1 - pad1;
    }

    result.n_in_data_tiles = std::min(params.n_inputs, searched_params.n_in_data_tiles);
    result.n_out_pix_tiles = std::min(params.n_outputs, searched_params.n_out_pix_tiles);

    // hacky fix of the incorrect kernel local memory address calculation for data
    result.out_pix_tile1 =
        (static_cast<int>(params.direction.IsForward()) == 0 && params.kernel_stride1 > 1)
            ? params.kernel_stride1
            : searched_params.out_pix_tile1;
    result.out_pix_tile0 =
        (static_cast<int>(params.direction.IsForward()) == 0 && params.kernel_stride0 > 1)
            ? params.kernel_stride0
            : searched_params.out_pix_tile0;

    result.out_pix_tile0 = result.out_pix_tile0 == 0 ? 1 : result.out_pix_tile0;
    result.out_pix_tile1 = result.out_pix_tile1 == 0 ? 1 : result.out_pix_tile1;

    result.grp_tile0 = std::max(8, (result.in_tile0 / result.out_pix_tile0));
    result.grp_tile1 = std::max(8, (result.in_tile1 / result.out_pix_tile1));
    result.in_tile0  = result.grp_tile0 * result.out_pix_tile0;
    result.in_tile1  = result.grp_tile1 * result.out_pix_tile1;

    int alu_tile0    = (result.in_tile0 + result.out_pix_tile0 - 1) / result.out_pix_tile0;
    int alu_tile1    = (result.in_tile1 + result.out_pix_tile1 - 1) / result.out_pix_tile1;
    int alu_tiles_sz = (alu_tile0 * alu_tile1);
    if(alu_tiles_sz > 256)
    {
        std::cerr << "ERROR: need out pix size ajustments\n";
        exit(0);
        // return ConvSolution(static_cast<miopenStatus_t>(-1));
    }

    int n_alus_total = (result.grp_tile0 * result.grp_tile1);

    result.n_stacks = std::min(result.n_stacks, (n_alus_total + alu_tiles_sz - 1) / alu_tiles_sz);
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);

    int n_alus_perstack = (n_alus_total + result.n_stacks - 1) / result.n_stacks;

    int n_read_procs;
    if((result.grp_tile1 * result.grp_tile0) <=
       static_cast<float>(result.in_tile1 * result.in_tile0))
    {
        n_read_procs = result.grp_tile1 * result.grp_tile0;
    }
    else
    {
        float proc_data_ratio = static_cast<float>(result.in_tile1 * result.in_tile0) /
                                static_cast<float>(result.grp_tile1 * result.grp_tile0);
        n_read_procs = (proc_data_ratio <= 0.25)
                           ? (result.grp_tile1 * result.grp_tile0) / 4
                           : (proc_data_ratio <= 0.5) ? (result.grp_tile1 * result.grp_tile0) / 2
                                                      : (result.grp_tile1 * result.grp_tile0);
    }

    int n_out_tile_blocks0 = (params.out_width + result.in_tile0 - 1) / (result.in_tile0);
    int n_out_tile_blocks1 = (params.out_height + result.in_tile1 - 1) / (result.in_tile1);

    int n_alu_tiles_perstack = (n_alus_perstack + alu_tiles_sz - 1) / alu_tiles_sz;
    int n_out_tiles_perstack = n_alu_tiles_perstack * result.n_out_pix_tiles;

    n_out_tiles_perstack = std::min(n_out_tiles_perstack, params.n_outputs);

    KernelInfo kernel_params;

    kernel_params.comp_options =
        std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(static_cast<long long>(hw_wave_sz)) +
        std::string(" -DMLO_DIR_FORWARD=") + (params.direction.IsForward() ? "1" : "0") +
        std::string(" -DMLO_FILTER_SIZE0=") +
        std::to_string(static_cast<long long>(params.kernel_size0)) +
        std::string(" -DMLO_FILTER_SIZE1=") +
        std::to_string(static_cast<long long>(params.kernel_size1)) +
        std::string(" -DMLO_FILTER_PAD0=") + std::to_string(static_cast<long long>(pad0)) +
        std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(pad1)) +
        std::string(" -DMLO_FILTER_STRIDE0=") +
        std::to_string(static_cast<long long>(params.kernel_stride0)) +
        std::string(" -DMLO_FILTER_STRIDE1=") +
        std::to_string(static_cast<long long>(params.kernel_stride1)) +
        std::string(" -DMLO_N_OUTPUTS=") +
        std::to_string(static_cast<long long>(params.n_outputs)) + std::string(" -DMLO_N_INPUTS=") +
        std::to_string(static_cast<long long>(params.n_inputs)) + std::string(" -DMLO_BATCH_SZ=") +
        std::to_string(static_cast<long long>(params.batch_sz)) + std::string(" -DMLO_OUT_WIDTH=") +
        std::to_string(static_cast<long long>(params.out_width)) +
        std::string(" -DMLO_OUT_HEIGHT=") +
        std::to_string(static_cast<long long>(params.out_height)) +
        std::string(" -DMLO_OUT_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(params.out_batch_stride)) +
        std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(params.out_channel_stride)) +
        std::string(" -DMLO_OUT_STRIDE=") +
        std::to_string(static_cast<long long>(params.out_stride)) +
        std::string(" -DMLO_IN_WIDTH=") + std::to_string(static_cast<long long>(params.in_width)) +
        std::string(" -DMLO_IN_HEIGHT=") +
        std::to_string(static_cast<long long>(params.in_height)) +
        std::string(" -DMLO_IN_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(params.in_batch_stride)) +
        std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(params.in_channel_stride)) +
        std::string(" -DMLO_IN_STRIDE=") + std::to_string(static_cast<long long>(params.in_stride))
        // algorithm parameters
        + std::string(" -DMLO_IN_TILE0=") +
        std::to_string(static_cast<long long>(result.in_tile0)) // size of input data per ALU plane
        + std::string(" -DMLO_IN_TILE1=") +
        std::to_string(static_cast<long long>(result.in_tile1)) // size of input data per ALU plane
        + std::string(" -DMLO_GRP_TILE0=") +
        std::to_string(static_cast<long long>(result.grp_tile0)) // # of ALUs (group size)
        + std::string(" -DMLO_GRP_TILE1=") +
        std::to_string(static_cast<long long>(result.grp_tile1)) //
        + std::string(" -DMLO_OUT_TILE0=") +
        std::to_string(
            static_cast<long long>(result.out_pix_tile0)) // size of ouptput tile per wk-item (ALU))
        + std::string(" -DMLO_OUT_TILE1=") +
        std::to_string(static_cast<long long>(result.out_pix_tile1)) //
        + std::string(" -DMLO_N_STACKS=") +
        std::to_string(static_cast<long long>(result.n_stacks)) // # of diff stacks (part of batch).
        + std::string(" -DMLO_N_OUT_TILES=") +
        std::to_string(static_cast<long long>(
            result.n_out_pix_tiles)) // # output pixel tiles per wk-item (ALU)
        + std::string(" -DMLO_N_OUT_TILES_PERSTACK=") +
        std::to_string(static_cast<long long>(n_out_tiles_perstack)) +
        std::string(" -DMLO_N_IN_TILES_PERSTACK=") +
        std::to_string(static_cast<long long>(
            result.n_in_data_tiles)) // total # of blocks of different inputs in LDS
        + std::string(" -DMLO_N_READ_PROCS=") +
        std::to_string(static_cast<long long>(n_read_procs)) + std::string(" -DMLO_CONV_BIAS=") +
        std::to_string(static_cast<long long>(params.bias)) + std::string(" -DMLO_ALU_VTILE0=") +
        std::to_string(static_cast<long long>(alu_tile0)) + std::string(" -DMLO_ALU_VTILE1=") +
        std::to_string(static_cast<long long>(alu_tile1)) + params.general_compile_options;

    kernel_params.l_wk.push_back(result.grp_tile1 * result.grp_tile0);
    kernel_params.l_wk.push_back(1);
    kernel_params.l_wk.push_back(1);

    size_t gbl_wk0 = n_out_tile_blocks0 * n_out_tile_blocks1;

    size_t gbl_wk1 = (params.n_outputs + n_out_tiles_perstack - 1) / n_out_tiles_perstack;
    size_t gbl_wk2 = (params.batch_sz + result.n_stacks - 1) / result.n_stacks;

    kernel_params.g_wk.push_back(gbl_wk0 * kernel_params.l_wk[0]);
    kernel_params.g_wk.push_back(gbl_wk1);
    kernel_params.g_wk.push_back(gbl_wk2);

    kernel_params.kernel_file = "MIOpenConvDirBatchNormActiv.cl";
    kernel_params.kernel_name = "MIOpenConvUniBatchNormActiv";

    // result.construction_params.push_back(kernel_params);
    return kernel_params;
}
}

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
                         int dilation_w)
{
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
    mlo_construct_direct2D construct_params(1); // forward

    construct_params.setOutputDescFromMLDesc(yDesc);
    construct_params.setInputDescFromMLDesc(xDesc);
    construct_params.setWeightDescFromMLDesc(wDesc);
    construct_params.setConvDescr(pad_h, pad_w, u, v, dilation_h, dilation_w);
    construct_params.setStream(&handle);

    mloConstruct(construct_params);
    std::string program_name       = construct_params.getKernelFile();
    std::string kernel_name        = construct_params.getKernelName();
    const std::string& parms       = construct_params.getCompilerOptions();
    const std::vector<size_t>& vld = construct_params.getLocalWkSize();
    const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);

    std::string algorithm_name = "miopenConvolutionFwdAlgoDirect";
    float padding_val          = 0;

    auto kernel = handle.AddKernel(
        algorithm_name, network_config, program_name, kernel_name, vld, vgd, parms);

    visit_float(xDesc.GetType(), [&](auto as_float) {
        {
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
    mlo_construct_direct2D construct_params(1); // forward

    construct_params.setOutputDescFromMLDesc(yDesc);
    construct_params.setInputDescFromMLDesc(xDesc);
    construct_params.setWeightDescFromMLDesc(wDesc);
    construct_params.setConvDescr(pad_h, pad_w, u, v, dilation_h, dilation_w);
    construct_params.setStream(&handle);

    ConvolutionContext params;
    construct_params.mloCopyTo(params);
    params.general_compile_options += " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";
    params.general_compile_options += " -DMIOPEN_NRN_OP_ID=" + std::to_string(activ_mode);
    if(bn_mode == miopenBNSpatial)
        params.general_compile_options += " -DSPATIAL_BN";
    else
        params.general_compile_options += " -DPERACT_BN";

    auto kernel_info               = solver::GetSolution(params);
    std::string program_name       = kernel_info.kernel_file;
    std::string kernel_name        = kernel_info.kernel_name;
    const std::string parms        = kernel_info.comp_options;
    const std::vector<size_t>& vld = kernel_info.l_wk;
    const std::vector<size_t>& vgd = kernel_info.g_wk;

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);
    network_config += std::to_string(activ_mode);

    std::string algorithm_name = "miopenConvolutionFwdAlgoDirect";
    float padding_val          = 0;

    auto kernel = handle.AddKernel(
        algorithm_name, network_config, program_name, kernel_name, vld, vgd, parms);

    visit_float(xDesc.GetType(), [&](auto as_float) {
        {
            auto f_activ_alpha = as_float(activ_alpha);
            auto f_activ_beta  = as_float(activ_beta);
            auto f_activ_gama  = as_float(activ_gama);

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
}
