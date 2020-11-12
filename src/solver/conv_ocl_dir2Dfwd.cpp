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

#include <miopen/handle.hpp>
#include <miopen/legacy_exhaustive_search.hpp>
#include <miopen/solver.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD)

namespace miopen {
namespace solver {

bool ConvOclDirectFwd::IsApplicable(const ConvolutionContext& params) const
{
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD{}))
        return false;
    if(!params.use_opencl_convolutions)
        return false;
    if(!params.Is2d())
        return false;
    if(params.IsAsymmetricPadH() || params.IsAsymmetricPadW())
        return false;
    if(!(params.IsFp32() || params.IsFp16() || params.IsBfp16()))
        return false;

    // clang-format off
    // Cases when dy has negative padding are not supported (issue 918)
    if(params.direction.IsBackwardData()
        && (params.GetBackwardPadW() < 0 || params.GetBackwardPadH() < 0))
        return false;

    // Factored out from ConvolutionDescriptor::IsDirectSupported(), which is now dissmissed:
    if (params.group_counts == 1)
    {
        const auto& p = params; //alias
        const bool supported =
            ((p.kernel_size_h == p.kernel_size_w)
              && (p.kernel_size_h == 3
                  || p.kernel_size_h == 5
                  || p.kernel_size_h == 7
                  || p.kernel_size_h == 9
                  || p.kernel_size_h == 11))
            || ((p.kernel_size_w == 10 || p.kernel_size_w == 20)
                && p.kernel_size_h == 5
                && p.kernel_stride_h == 2
                && p.kernel_stride_w == 2
                && p.pad_h == 0
                && p.pad_w == 0)
            /// The following is for #1594. Most likely we can open more configs,
            /// but that would require thorough testing.
            || (p.IsFp16()
                && p.kernel_size_h == 4
                && p.kernel_size_w == 4
                && p.pad_h == 0
                && p.pad_w == 0);

        if (!supported)
            return false;
    }
    return params.kernel_stride_w == params.kernel_stride_h
        && params.pad_w == params.pad_h
        && params.kernel_dilation_w == 1
        && params.kernel_dilation_h == 1
        /// \todo need to make sure support stride > 2, should support but not tested
        && !(params.kernel_stride_w > 2 || params.kernel_stride_h > 2)
        /// We have optimized 1x1 kernel for normal conv.
        && !(params.group_counts == 1 && params.kernel_size_h == 1 && params.kernel_size_w == 1)
        /// \todo Workaround to avoid FP16 precision issue:
        /// While MIOpenConvUni is up to 4x faster than MIOpenCDFGen (even not auto-tuned),
        /// it seems that is has 4x..20x worse precision, and some "test_conv --half" tests fail.
        /// See issue #1626.
        && !(params.direction.IsForward()
            && params.IsFp16()
            && params.kernel_stride_w == 2)
        && IsValidPerformanceConfig(params, GetPerformanceConfig(params));
    // clang-format on
}

/// This prevents errors in ConvOclDirectFwd::GetSolution(),
/// or during OpenCL build. Reproduces logic from GetSolution()
/// and some logic from the corresponding opencl kernel source.
/// The cases which lead to errors can be later omitted from the search.
/// \todo Get rid the duplication of code where possible.
bool ConvOclDirectFwd::IsValidPerformanceConfig(
    const ConvolutionContext& params, const LegacyPerformanceConfig& searched_params) const
{
    ConvSolution result;

    searched_params.CopyTo(result);
    // auto pad_w = params.pad_w;
    // auto pad_h = params.pad_h;
    // auto hw_wave_sz = 64;
    // if(!params.direction.IsForward())
    // {
    //     // backward
    //     pad_w = params.kernel_size_w - 1 - pad_w;
    //     pad_h = params.kernel_size_h - 1 - pad_h;
    // }
    auto group_counts = params.group_counts;
    result.n_in_data_tiles =
        std::min(params.n_inputs / group_counts, searched_params.n_in_data_tiles);
    result.n_out_pix_tiles =
        std::min(params.n_outputs / group_counts, searched_params.n_out_pix_tiles);

    // hacky fix of the incorrect kernel local memory address calculation for data
    result.out_pix_tile1 = (!params.direction.IsForward() && params.kernel_stride_h > 1)
                               ? params.kernel_stride_h
                               : searched_params.out_pix_tile1;
    result.out_pix_tile0 = (!params.direction.IsForward() && params.kernel_stride_w > 1)
                               ? params.kernel_stride_w
                               : searched_params.out_pix_tile0;

    if(result.out_pix_tile1 == 0 || result.out_pix_tile0 == 0 /* DIV/0 */)
    {
        return false;
    }

    result.grp_tile0 = std::max(8, (result.in_tile0 / result.out_pix_tile0));
    result.grp_tile1 = std::max(8, (result.in_tile1 / result.out_pix_tile1));
    result.in_tile0  = result.grp_tile0 * result.out_pix_tile0;
    result.in_tile1  = result.grp_tile1 * result.out_pix_tile1;

    int alu_tile0    = (result.in_tile0 + result.out_pix_tile0 - 1) / result.out_pix_tile0;
    int alu_tile1    = (result.in_tile1 + result.out_pix_tile1 - 1) / result.out_pix_tile1;
    int alu_tiles_sz = (alu_tile0 * alu_tile1);
    if(alu_tiles_sz > 256 || alu_tiles_sz == 0 /* DIV/0 */)
    {
        return false;
    }

    int n_alus_total = (result.grp_tile0 * result.grp_tile1);

    result.n_stacks = std::min(result.n_stacks, (n_alus_total + alu_tiles_sz - 1) / alu_tiles_sz);
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);

    if(result.n_stacks == 0 /* DIV/0 */)
    {
        return false;
    }
    int n_alus_perstack = (n_alus_total + result.n_stacks - 1) / result.n_stacks;

    // int n_read_procs;
    // if((result.grp_tile1 * result.grp_tile0) <= static_cast<float>(result.in_tile1 *
    // result.in_tile0))
    // {
    //     n_read_procs = result.grp_tile1 * result.grp_tile0;
    // }
    // else
    // {
    //     float proc_data_ratio = static_cast<float>(result.in_tile1 * result.in_tile0) /
    //                             static_cast<float>(result.grp_tile1 * result.grp_tile0);
    //     n_read_procs = (proc_data_ratio <= 0.25)
    //                        ? (result.grp_tile1 * result.grp_tile0) / 4
    //                        : (proc_data_ratio <= 0.5) ? (result.grp_tile1 * result.grp_tile0) / 2
    //                                                   : (result.grp_tile1 * result.grp_tile0);
    // }

    // int n_out_tile_blocks0 = (params.out_width + result.in_tile0 - 1) / (result.in_tile0);
    // int n_out_tile_blocks1 = (params.out_height + result.in_tile1 - 1) / (result.in_tile1);
    int n_alu_tiles_perstack = (n_alus_perstack + alu_tiles_sz - 1) / alu_tiles_sz;
    int n_out_tiles_perstack = n_alu_tiles_perstack * result.n_out_pix_tiles;
    n_out_tiles_perstack     = std::min(n_out_tiles_perstack, params.n_outputs / group_counts);

    // const auto mlo_hw_wave_sz=hw_wave_sz;
    const auto mlo_filter_size0 = static_cast<long long>(params.kernel_size_w);
    const auto mlo_filter_size1 = static_cast<long long>(params.kernel_size_h);
    // const auto mlo_filter_pad0=static_cast<long long>(pad_w);
    // const auto mlo_filter_pad1=static_cast<long long>(pad_h);
    const auto mlo_filter_stride0 = static_cast<long long>(params.kernel_stride_w);
    const auto mlo_filter_stride1 = static_cast<long long>(params.kernel_stride_h);
    // const auto mlo_n_outputs=static_cast<long long>(params.n_outputs);
    // const auto mlo_n_inputs=static_cast<long long>(params.n_inputs);
    // const auto mlo_batch_sz=static_cast<long long>(params.batch_sz);
    // const auto mlo_out_width=static_cast<long long>(params.out_width);
    // const auto mlo_out_height=static_cast<long long>(params.out_height);
    // const auto mlo_out_batch_stride=static_cast<long long>(params.out_batch_stride);
    // const auto mlo_out_channel_stride=static_cast<long long>(params.out_channel_stride);
    // const auto mlo_out_stride=static_cast<long long>(params.out_stride);
    // const auto mlo_in_width=static_cast<long long>(params.in_width);
    // const auto mlo_in_height=static_cast<long long>(params.in_height);
    // const auto mlo_in_batch_stride=static_cast<long long>(params.in_batch_stride);
    // const auto mlo_in_channel_stride=static_cast<long long>(params.in_channel_stride);
    // const auto mlo_in_stride=static_cast<long long>(params.in_stride);
    // algorithm parameters
    const auto mlo_in_tile0 = static_cast<long long>(result.in_tile0);
    const auto mlo_in_tile1 = static_cast<long long>(result.in_tile1);
    // const auto mlo_grp_tile0=static_cast<long long>(result.grp_tile0);
    // const auto mlo_grp_tile1=static_cast<long long>(result.grp_tile1);
    // const auto mlo_out_tile0=static_cast<long long>(result.out_pix_tile0);
    // const auto mlo_out_tile1=static_cast<long long>(result.out_pix_tile1);
    const auto mlo_n_stacks = static_cast<long long>(result.n_stacks);
    // const auto mlo_n_out_tiles=static_cast<long long>(result.n_out_pix_tiles);
    const auto mlo_n_out_tiles_perstack = static_cast<long long>(n_out_tiles_perstack);
    const auto mlo_n_in_tiles_perstack  = static_cast<long long>(result.n_in_data_tiles);
    // const auto mlo_n_read_procs=static_cast<long long>(n_read_procs);
    // const auto mlo_conv_bias=static_cast<long long>(params.bias);
    // const auto mlo_alu_vtile0=static_cast<long long>(alu_tile0);
    // const auto mlo_alu_vtile1=static_cast<long long>(alu_tile1);

    if(n_out_tiles_perstack == 0 /* DIV/0 */)
    {
        return false;
    }

    // Reproduces some build logic (preprocessing computations) from the opencl source.
    // Variables whose names begin with "mlo_" represent corresponding "MLO_" macros,
    // e.g. mlo_in_lcl_width here represents MLO_IN_LCL_WIDTH macro in the opencl source.
    long long mlo_in_lcl_width;
    long long mlo_in_lcl_height;
    if(params.direction.IsForward())
    {
        mlo_in_lcl_width  = ((mlo_in_tile0 - 1) * mlo_filter_stride0 + mlo_filter_size0);
        mlo_in_lcl_height = ((mlo_in_tile1 - 1) * mlo_filter_stride1 + mlo_filter_size1);
    }
    else
    {
        assert(mlo_filter_stride0 != 0 && mlo_filter_stride1 != 0);
        mlo_in_lcl_width =
            ((mlo_in_tile0 + mlo_filter_size0 - 1 + mlo_filter_stride0 - 1) / mlo_filter_stride0);
        mlo_in_lcl_height =
            ((mlo_in_tile1 + mlo_filter_size1 - 1 + mlo_filter_stride1 - 1) / mlo_filter_stride1);
    }
    const auto mlo_in_lcl_tile_sz     = (mlo_in_lcl_width * mlo_in_lcl_height);
    const auto mlo_in_lcl_perstack_sz = (mlo_in_lcl_tile_sz * mlo_n_in_tiles_perstack);
    const auto mlo_in_lcl_sz          = (mlo_in_lcl_perstack_sz * mlo_n_stacks);
    const auto mlo_filter_sz          = (mlo_filter_size1 * mlo_filter_size0);
    const auto mlo_weights_sz =
        (mlo_n_out_tiles_perstack * mlo_n_in_tiles_perstack * mlo_filter_sz);
    const auto sizeof_float     = GetTypeSize(params.in_data_type);
    const auto mlo_lds_max_size = 65536;
    //    MIOPEN_LOG_I("((mlo_in_lcl_sz + mlo_weights_sz) * sizeof_float)=" << ((mlo_in_lcl_sz +
    //    mlo_weights_sz) * sizeof_float));
    if(((mlo_in_lcl_sz + mlo_weights_sz) * sizeof_float) > mlo_lds_max_size)
    {
        return false; // NOLINT
    }
    return true;
}

inline ConvSolution BaseGetSolution(const ConvolutionContext& params,
                                    const LegacyPerformanceConfig& searched_params)
{
    ConvSolution result;

    // std::size_t localMemSize = params.stream.GetLocalMemorySize();

    searched_params.CopyTo(result);
    auto pad_w        = params.pad_w;
    auto pad_h        = params.pad_h;
    auto group_counts = params.group_counts;
    auto hw_wave_sz   = 64;
    // auto dev_local_mem_sz = localMemSize; // in bytes

    if(!params.direction.IsForward())
    {
        // backward
        pad_w = params.kernel_size_w - 1 - pad_w;
        pad_h = params.kernel_size_h - 1 - pad_h;
    }

    result.n_in_data_tiles =
        std::min(params.n_inputs / group_counts, searched_params.n_in_data_tiles);
    result.n_out_pix_tiles =
        std::min(params.n_outputs / group_counts, searched_params.n_out_pix_tiles);

    // hacky fix of the incorrect kernel local memory address calculation for data
    result.out_pix_tile1 = (!params.direction.IsForward() && params.kernel_stride_h > 1)
                               ? params.kernel_stride_h
                               : searched_params.out_pix_tile1;
    result.out_pix_tile0 = (!params.direction.IsForward() && params.kernel_stride_w > 1)
                               ? params.kernel_stride_w
                               : searched_params.out_pix_tile0;

    if(result.out_pix_tile1 == 0 || result.out_pix_tile0 == 0 /* DIV/0 */)
    {
        MIOPEN_LOG_E("result.out_pix_tile1 == 0 || result.out_pix_tile0 == 0");
        return ConvSolution(miopenStatusInternalError);
    }
    result.grp_tile0 = std::max(8, (result.in_tile0 / result.out_pix_tile0));
    result.grp_tile1 = std::max(8, (result.in_tile1 / result.out_pix_tile1));
    result.in_tile0  = result.grp_tile0 * result.out_pix_tile0;
    result.in_tile1  = result.grp_tile1 * result.out_pix_tile1;

    int alu_tile0    = (result.in_tile0 + result.out_pix_tile0 - 1) / result.out_pix_tile0;
    int alu_tile1    = (result.in_tile1 + result.out_pix_tile1 - 1) / result.out_pix_tile1;
    int alu_tiles_sz = (alu_tile0 * alu_tile1);
    if(alu_tiles_sz > 256 || alu_tiles_sz == 0 /* DIV/0 */)
    {
        MIOPEN_LOG_E("need out pix size ajustments (alu_tiles_sz > 256 || alu_tiles_sz == 0)");
        return ConvSolution(miopenStatusInternalError);
    }

    int n_alus_total = (result.grp_tile0 * result.grp_tile1);

    result.n_stacks = std::min(result.n_stacks, (n_alus_total + alu_tiles_sz - 1) / alu_tiles_sz);
    result.n_stacks = std::min(params.batch_sz, result.n_stacks);

    if(result.n_stacks == 0 /* DIV/0 */)
    {
        MIOPEN_LOG_E("result.n_stacks == 0");
        return ConvSolution(miopenStatusInternalError);
    }
    int n_alus_perstack = (n_alus_total + result.n_stacks - 1) / result.n_stacks;

    int n_read_procs;
    if((result.grp_tile1 * result.grp_tile0) <= (result.in_tile1 * result.in_tile0))
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

    n_out_tiles_perstack = std::min(n_out_tiles_perstack, params.n_outputs / group_counts);

    KernelInfo kernel_params;

    kernel_params.comp_options =
        std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(static_cast<long long>(hw_wave_sz)) +
        std::string(" -DMLO_DIR_FORWARD=") + (params.direction.IsForward() ? "1" : "0") +
        std::string(" -DMLO_FILTER_SIZE0=") +
        std::to_string(static_cast<long long>(params.kernel_size_w)) +
        std::string(" -DMLO_FILTER_SIZE1=") +
        std::to_string(static_cast<long long>(params.kernel_size_h)) +
        std::string(" -DMLO_FILTER_PAD0=") + std::to_string(static_cast<long long>(pad_w)) +
        std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(pad_h)) +
        std::string(" -DMLO_FILTER_STRIDE0=") +
        std::to_string(static_cast<long long>(params.kernel_stride_w)) +
        std::string(" -DMLO_FILTER_STRIDE1=") +
        std::to_string(static_cast<long long>(params.kernel_stride_h)) +
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
        std::string(" -DMLO_IN_STRIDE=") +
        std::to_string(static_cast<long long>(params.in_stride))
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
        std::to_string(static_cast<long long>(n_read_procs)) + std::string(" -DMLO_ALU_VTILE0=") +
        std::to_string(static_cast<long long>(alu_tile0)) + std::string(" -DMLO_ALU_VTILE1=") +
        std::to_string(static_cast<long long>(alu_tile1)) + params.general_compile_options;

    if(group_counts >= 2)
    {
        kernel_params.comp_options += (std::string(" -DMLO_GROUP_COUNTS=") +
                                       std::to_string(static_cast<long long>(group_counts)));
        kernel_params.comp_options +=
            (std::string(" -DMLO_GROUP_TILES=") +
             std::to_string(static_cast<long long>(params.n_outputs / group_counts)));
        kernel_params.comp_options +=
            (std::string(" -DMLO_STACK_PERGROUP=") +
             std::to_string(static_cast<long long>(
                 (params.n_outputs / group_counts + n_out_tiles_perstack - 1) /
                 n_out_tiles_perstack)));
        kernel_params.comp_options += std::string(" -DGRP_MOD_ENABLE");
    }

    kernel_params.l_wk.push_back(result.grp_tile1 * result.grp_tile0);
    kernel_params.l_wk.push_back(1);
    kernel_params.l_wk.push_back(1);

    size_t gbl_wk0 = n_out_tile_blocks0 * n_out_tile_blocks1;

    if(n_out_tiles_perstack == 0 /* DIV/0 */)
    {
        MIOPEN_LOG_E("n_out_tiles_perstack == 0");
        return ConvSolution(miopenStatusInternalError);
    }
    size_t gbl_wk1 = group_counts >= 2
                         ? (((params.n_outputs / group_counts + n_out_tiles_perstack - 1) /
                             n_out_tiles_perstack) *
                            group_counts)
                         : ((params.n_outputs + n_out_tiles_perstack - 1) / n_out_tiles_perstack);
    size_t gbl_wk2 = (params.batch_sz + result.n_stacks - 1) / result.n_stacks;

    kernel_params.g_wk.push_back(gbl_wk0 * kernel_params.l_wk[0]);
    kernel_params.g_wk.push_back(gbl_wk1);
    kernel_params.g_wk.push_back(gbl_wk2);

    kernel_params.kernel_file = "MIOpenConvDirUni.cl";
    kernel_params.kernel_name = "MIOpenConvUni";

    result.construction_params.push_back(kernel_params);
    result.invoker_factory = &conv::MakeGenericXWYPadInvoker;

    if(params.direction.IsForward())
        result.invoker_factory = &conv::MakeGenericXWYPadInvoker;
    std::ostringstream os_params;
    searched_params.Serialize(os_params);
    result.performance_config = os_params.str();
    return result;
}

std::vector<ConvSolution> ConvOclDirectFwd::GetSolutions(const ConvolutionContext& params,
                                                         const bool onlyGetDefault) const
{
    this->GetSolutions(*this, params, onlyGetDefault);
}

std::vector<ConvSolution> ConvOclDirectFwd::GetSolutions(Solver solver,
                                                         const ConvolutionContext& params,
                                                         const bool onlyGetDefault) const
{
    std::vector<ConvSolution> all_solutions;
    size_t solutions_counter = 0, failed_counter = 0;
    const auto default_config = GetPerformanceConfig(params);
    if(this->IsValidPerformanceConfig(params, default_config))
    {
        auto default_solution = solver.GetSolution(params, default_config);
        if(default_solution.Succeeded())
        {
            ++solutions_counter;
            all_solutions.push_back(default_solution);
        }
        else
        {
            ++failed_counter;
        }
        
    }
    if(onlyGetDefault)
    {
        return all_solutions;
    }

    // get loop here
    int tile_sz1[4]        = {8, 16, 32, 64};
    int tile_sz0[4]        = {8, 16, 32, 64};
    int out_pix_tile_sz[3] = {1, 2, 4};
    int n_out_tiles_rg[5]  = {1, 2, 4, 8};
    int n_in_tiles_rg[3]   = {1, 2, 4};
    int n_in_stacks_sz[2]  = {1, 2};

    int out_pix_tl_cnt = 3; // out_pix_tile_sz[1];
    int n_out_tls      = 4;
    int n_in_tls       = 3;
    int stack_cnt      = std::min(params.batch_sz, 2);
    int n_tile0_sz     = 4;
    int n_tile1_sz     = 4;

    if(params.out_width >= 16)
    {
        tile_sz0[0] = 16;
        tile_sz0[1] = 32;
        n_tile0_sz  = 2;
    }

    if(params.out_height >= 16)
    {
        tile_sz1[0] = 16;
        tile_sz1[1] = 32;
        n_tile1_sz  = 2;
    }

    int n_tiles_cnt = n_tile0_sz * n_tile1_sz;

    long long total_configs = 0;

    LegacyPerformanceConfig current_config;
    total_configs = n_tiles_cnt * out_pix_tl_cnt * out_pix_tl_cnt * n_out_tls *
             n_in_tls * stack_cnt + 1;
    MIOPEN_LOG_I2("Get all " << total_configs << " Solutions in the 9 dim space.");
    // tile 1
    for(int j = 0; j < n_tile1_sz; ++j)
    {
        int tile_sz[3]  = {8, 16, 32};
        current_config.in_tile1 = tile_sz1[j];
        if(params.out_height * 2 <= current_config.in_tile1 && current_config.in_tile1 > tile_sz[0])
        {
            continue;
        }

        // tile 0
        for(int i = 0; i < n_tile0_sz; ++i)
        {
            current_config.in_tile0 = tile_sz0[i];
            if((params.out_width * 2 <= current_config.in_tile0 && current_config.in_tile0 > tile_sz[0]))
            {
                continue;
            }
            if(params.out_height > 16 && params.out_width > 16 &&
               ((current_config.in_tile1 == 8 && current_config.in_tile0 == 8) ||
                (current_config.grp_tile0 == 8 && current_config.grp_tile1 == 8)))
            {
                continue;
            }
            if(params.out_width > 32 && current_config.in_tile1 > current_config.in_tile0)
            {
                continue;
            }
            // out pix 1

            for(int k = 0; k < out_pix_tl_cnt; ++k)
            {
                current_config.out_pix_tile1 = out_pix_tile_sz[k];
                current_config.grp_tile1     = current_config.in_tile1 / current_config.out_pix_tile1;
                if(current_config.out_pix_tile1 > current_config.in_tile1 || current_config.grp_tile1 < 8)
                {
                    continue;
                }
                // out pix 0

                for(int l = 0; l < out_pix_tl_cnt; ++l)
                {
                    current_config.out_pix_tile0 = out_pix_tile_sz[l];
                    current_config.grp_tile0     = current_config.in_tile0 / current_config.out_pix_tile0;

                    if(current_config.out_pix_tile0 > current_config.in_tile0 || current_config.grp_tile0 < 8)
                    {
                        continue;
                    }

                    for(int o_t = 0; o_t < n_out_tls; ++o_t)
                    {
                        current_config.n_out_pix_tiles = n_out_tiles_rg[o_t];
                        if(params.n_outputs < current_config.n_out_pix_tiles)
                        {
                            continue;
                        }

                        for(int i_t = 0; i_t < n_in_tls; ++i_t)
                        {
                            current_config.n_in_data_tiles = n_in_tiles_rg[i_t];
                            if(params.n_inputs < current_config.n_in_data_tiles)
                            {
                                continue;
                            }

                            for(int s = 0; s < stack_cnt; ++s)
                            {

                                current_config.n_stacks = n_in_stacks_sz[s];
                                if(current_config.n_stacks > params.batch_sz)
                                {
                                    continue;
                                }

                                if(current_config.out_pix_tile1 * current_config.out_pix_tile0 *
                                       current_config.n_out_pix_tiles * current_config.n_stacks >=
                                    128)
                                {
                                    continue;
                                }
                                if(this->IsValidPerformanceConfig(params, current_config))
                                {
                                    auto currentSolution = solver.GetSolution(params, current_config);
                                    if(currentSolution.Succeeded())
                                    {
                                        ++solutions_counter;
                                        all_solutions.push_back(currentSolution);
                                    }
                                    else
                                    {
                                        ++failed_counter;
                                    }
                                }
                                else
                                {
                                    ++failed_counter;
                                }
                                MIOPEN_LOG_I2("##(n_get, n_failed, n_total): "
                                             << solutions_counter << " / " << failed_counter << " / "
                                             << total_configs << ", "
                                             << current_config);
                            }
                        }
                    }
                }
            }
        }
    }
    return all_solutions;
}

ConvSolution ConvOclDirectFwd::GetSolution(const ConvolutionContext& params,
                                           const LegacyPerformanceConfig& searched_params) const
{
    ConvSolution result = BaseGetSolution(params, searched_params);

    if(result.Succeeded())
    {
        result.construction_params[0].comp_options +=
            std::string(" -DMLO_CONV_BIAS=") + std::to_string(static_cast<long long>(params.bias)) +
            params.general_compile_options;
    }

    return result;
}

std::vector<ConvSolution>
ConvOclDirectFwdFused::GetSolutions(const ConvolutionContext& params,
                                    const bool onlyGetDefault) const
{
    ConvOclDirectFwd::GetSolutions(*this, params, onlyGetDefault);
}

ConvSolution
ConvOclDirectFwdFused::GetSolution(const ConvolutionContext& params,
                                   const LegacyPerformanceConfig& searched_params) const
{
    ConvSolution result = BaseGetSolution(params, searched_params);
    if(result.Succeeded())
    {
        result.construction_params[0].comp_options += params.general_compile_options;
    }
    return result;
}
} // namespace solver
} // namespace miopen
