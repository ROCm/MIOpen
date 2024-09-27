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
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvOclDirectFwd::IsApplicable(const ExecutionContext& ctx,
                                    const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_OCL_FWD))
        return false;
    if(ThisSolverIsDeprecatedStatic::IsDisabled(ctx))
        return false;
    if(!ctx.use_opencl_convolutions)
        return false;
    if(!problem.Is2d())
        return false;
    if(!(problem.IsDirectionForward() || problem.IsDirectionBackwardData()))
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16()))
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsLayoutDefault())
        return false;

    // clang-format off
    // Cases when dy has negative padding are not supported (issue 918)
    if(problem.IsDirectionBackwardData()
        && (problem.GetBackwardPadW() < 0 || problem.GetBackwardPadH() < 0))
        return false;

    // Factored out from ConvolutionDescriptor::IsDirectSupported(), which is now dissmissed:
    if (problem.GetGroupCount() == 1)
    {
        const auto& p = problem; //alias
        const bool supported =
            ((p.GetWeightsHeight() == p.GetWeightsWidth())
              && (p.GetWeightsHeight() == 3
                  || p.GetWeightsHeight() == 5
                  || p.GetWeightsHeight() == 7
                  || p.GetWeightsHeight() == 9
                  || p.GetWeightsHeight() == 11))
            || ((p.GetWeightsWidth() == 10 || p.GetWeightsWidth() == 20)
                && p.GetWeightsHeight() == 5
                && p.GetKernelStrideH() == 2
                && p.GetKernelStrideW() == 2
                && p.GetPadH() == 0
                && p.GetPadW() == 0)
            /// The following is for #1594. Most likely we can open more configs,
            /// but that would require thorough testing.
            || (p.IsFp16()
                && p.GetWeightsHeight() == 4
                && p.GetWeightsWidth() == 4
                && p.GetPadH() == 0
                && p.GetPadW() == 0);

        if (!supported)
            return false;
    }
    return problem.GetKernelStrideW() == problem.GetKernelStrideH()
        && problem.GetPadW() == problem.GetPadH()
        && problem.GetDilationW() == 1
        && problem.GetDilationH() == 1
        /// \todo need to make sure support stride > 2, should support but not tested
        && !(problem.GetKernelStrideW() > 2 || problem.GetKernelStrideH() > 2)
        /// We have optimized 1x1 kernel for normal conv.
        && !(problem.GetGroupCount() == 1 && problem.GetWeightsHeight() == 1 && problem.GetWeightsWidth() == 1)
        /// \todo Workaround to avoid FP16 precision issue:
        /// While MIOpenConvUni is up to 4x faster than MIOpenCDFGen (even not auto-tuned),
        /// it seems that is has 4x..20x worse precision, and some "test_conv --half" tests fail.
        /// See issue #1626.
        && !(problem.IsDirectionForward()
            && problem.IsFp16()
            && problem.GetKernelStrideW() == 2)
        && IsValidPerformanceConfig(ctx, problem, GetDefaultPerformanceConfig(ctx, problem));
    // clang-format on
}

/// This prevents errors in ConvOclDirectFwd::GetSolution(),
/// or during OpenCL build. Reproduces logic from GetSolution()
/// and some logic from the corresponding opencl kernel source.
/// The cases which lead to errors can be later omitted from the search.
/// \todo Get rid the duplication of code where possible.
bool ConvOclDirectFwd::IsValidPerformanceConfig(const ExecutionContext&,
                                                const ProblemDescription& problem,
                                                const LegacyPerformanceConfig& config) const
{
    ConvSolution result;

    config.CopyTo(result);
    // auto pad_w = problem.GetPadW();
    // auto pad_h = problem.GetPadH();
    // auto hw_wave_sz = 64;
    // if(!problem.IsDirectionForward())
    // {
    //     // backward
    //     pad_w = problem.GetBackwardPadW();
    //     pad_h = problem.GetBackwardPadH();
    // }
    auto group_counts = problem.GetGroupCount();
    result.n_in_data_tiles =
        std::min(static_cast<int>(problem.GetInChannels()) / group_counts, config.n_in_data_tiles);
    result.n_out_pix_tiles =
        std::min(static_cast<int>(problem.GetOutChannels()) / group_counts, config.n_out_pix_tiles);

    // hacky fix of the incorrect kernel local memory address calculation for data
    result.out_pix_tile1 = (!problem.IsDirectionForward() && problem.GetKernelStrideH() > 1)
                               ? problem.GetKernelStrideH()
                               : config.out_pix_tile1;
    result.out_pix_tile0 = (!problem.IsDirectionForward() && problem.GetKernelStrideW() > 1)
                               ? problem.GetKernelStrideW()
                               : config.out_pix_tile0;

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
    result.n_stacks = std::min(static_cast<int>(problem.GetBatchSize()), result.n_stacks);

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

    // int n_out_tile_blocks0 = (problem.GetOutWidth() + result.in_tile0 - 1) / (result.in_tile0);
    // int n_out_tile_blocks1 = (problem.GetOutHeight() + result.in_tile1 - 1) / (result.in_tile1);
    int n_alu_tiles_perstack = (n_alus_perstack + alu_tiles_sz - 1) / alu_tiles_sz;
    int n_out_tiles_perstack = n_alu_tiles_perstack * result.n_out_pix_tiles;
    n_out_tiles_perstack =
        std::min(n_out_tiles_perstack, static_cast<int>(problem.GetOutChannels()) / group_counts);

    // const auto mlo_hw_wave_sz=hw_wave_sz;
    const auto mlo_filter_size0 = static_cast<long long>(problem.GetWeightsWidth());
    const auto mlo_filter_size1 = static_cast<long long>(problem.GetWeightsHeight());
    // const auto mlo_filter_pad0=static_cast<long long>(pad_w);
    // const auto mlo_filter_pad1=static_cast<long long>(pad_h);
    const auto mlo_filter_stride0 = static_cast<long long>(problem.GetKernelStrideW());
    const auto mlo_filter_stride1 = static_cast<long long>(problem.GetKernelStrideH());
    // const auto mlo_n_outputs=static_cast<long long>(problem.GetOutChannels());
    // const auto mlo_n_inputs=static_cast<long long>(problem.GetInChannels());
    // const auto mlo_batch_sz=static_cast<long long>(problem.GetBatchSize());
    // const auto mlo_out_width=static_cast<long long>(problem.GetOutWidth());
    // const auto mlo_out_height=static_cast<long long>(problem.GetOutHeight());
    // const auto mlo_out_batch_stride=static_cast<long long>(problem.GetOutBatchStride());
    // const auto mlo_out_channel_stride=static_cast<long long>(problem.GetOutChannelStride());
    // const auto mlo_out_stride=static_cast<long long>(problem.GetOutStrideH());
    // const auto mlo_in_width=static_cast<long long>(problem.GetInWidth());
    // const auto mlo_in_height=static_cast<long long>(problem.GetInHeight());
    // const auto mlo_in_batch_stride=static_cast<long long>(problem.GetInBatchStride());
    // const auto mlo_in_channel_stride=static_cast<long long>(problem.GetInChannelStride());
    // const auto mlo_in_stride=static_cast<long long>(problem.GetInStrideH());
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
    // const auto mlo_conv_bias=static_cast<long long>(problem.GetBias());
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
    if(problem.IsDirectionForward())
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
    const auto sizeof_float     = GetTypeSize(problem.GetInDataType());
    const auto mlo_lds_max_size = 65536;
    //    MIOPEN_LOG_I("((mlo_in_lcl_sz + mlo_weights_sz) * sizeof_float)=" << ((mlo_in_lcl_sz +
    //    mlo_weights_sz) * sizeof_float));
    if(((mlo_in_lcl_sz + mlo_weights_sz) * sizeof_float) > mlo_lds_max_size)
    {
        return false; // NOLINT
    }
    return true;
}

ConvSolution ConvOclDirectFwd::BaseGetSolution(const ExecutionContext& ctx,
                                               const ProblemDescription& problem,
                                               const LegacyPerformanceConfig& config)
{
    ConvSolution result;

    // std::size_t localMemSize = ctx.stream.GetLocalMemorySize();

    config.CopyTo(result);
    auto pad_w        = problem.GetPadW();
    auto pad_h        = problem.GetPadH();
    auto group_counts = problem.GetGroupCount();
    auto hw_wave_sz   = 64;
    // auto dev_local_mem_sz = localMemSize; // in bytes

    if(!problem.IsDirectionForward())
    {
        // backward
        pad_w = problem.GetBackwardPadW();
        pad_h = problem.GetBackwardPadH();
    }

    result.n_in_data_tiles =
        std::min(static_cast<int>(problem.GetInChannels()) / group_counts, config.n_in_data_tiles);
    result.n_out_pix_tiles =
        std::min(static_cast<int>(problem.GetOutChannels()) / group_counts, config.n_out_pix_tiles);

    // hacky fix of the incorrect kernel local memory address calculation for data
    result.out_pix_tile1 = (!problem.IsDirectionForward() && problem.GetKernelStrideH() > 1)
                               ? problem.GetKernelStrideH()
                               : config.out_pix_tile1;
    result.out_pix_tile0 = (!problem.IsDirectionForward() && problem.GetKernelStrideW() > 1)
                               ? problem.GetKernelStrideW()
                               : config.out_pix_tile0;

    if(result.out_pix_tile1 == 0 || result.out_pix_tile0 == 0 /* DIV/0 */)
    {
        MIOPEN_LOG_E("result.out_pix_tile1 == 0 || result.out_pix_tile0 == 0");
        return {miopenStatusInternalError};
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
        return {miopenStatusInternalError};
    }

    int n_alus_total = (result.grp_tile0 * result.grp_tile1);

    result.n_stacks = std::min(result.n_stacks, (n_alus_total + alu_tiles_sz - 1) / alu_tiles_sz);
    result.n_stacks = std::min(static_cast<int>(problem.GetBatchSize()), result.n_stacks);

    if(result.n_stacks == 0 /* DIV/0 */)
    {
        MIOPEN_LOG_E("result.n_stacks == 0");
        return {miopenStatusInternalError};
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
        n_read_procs = (proc_data_ratio <= 0.25)  ? (result.grp_tile1 * result.grp_tile0) / 4
                       : (proc_data_ratio <= 0.5) ? (result.grp_tile1 * result.grp_tile0) / 2
                                                  : (result.grp_tile1 * result.grp_tile0);
    }

    int n_out_tile_blocks0 = (problem.GetOutWidth() + result.in_tile0 - 1) / (result.in_tile0);
    int n_out_tile_blocks1 = (problem.GetOutHeight() + result.in_tile1 - 1) / (result.in_tile1);

    int n_alu_tiles_perstack = (n_alus_perstack + alu_tiles_sz - 1) / alu_tiles_sz;
    int n_out_tiles_perstack = n_alu_tiles_perstack * result.n_out_pix_tiles;

    n_out_tiles_perstack =
        std::min(n_out_tiles_perstack, static_cast<int>(problem.GetOutChannels()) / group_counts);

    KernelInfo kernel_params;

    kernel_params.comp_options =
        std::string(" -DMLO_HW_WAVE_SZ=") + std::to_string(static_cast<long long>(hw_wave_sz)) +
        std::string(" -DMLO_DIR_FORWARD=") + (problem.IsDirectionForward() ? "1" : "0") +
        std::string(" -DMLO_FILTER_SIZE0=") +
        std::to_string(static_cast<long long>(problem.GetWeightsWidth())) +
        std::string(" -DMLO_FILTER_SIZE1=") +
        std::to_string(static_cast<long long>(problem.GetWeightsHeight())) +
        std::string(" -DMLO_FILTER_PAD0=") + std::to_string(static_cast<long long>(pad_w)) +
        std::string(" -DMLO_FILTER_PAD1=") + std::to_string(static_cast<long long>(pad_h)) +
        std::string(" -DMLO_FILTER_STRIDE0=") +
        std::to_string(static_cast<long long>(problem.GetKernelStrideW())) +
        std::string(" -DMLO_FILTER_STRIDE1=") +
        std::to_string(static_cast<long long>(problem.GetKernelStrideH())) +
        std::string(" -DMLO_N_OUTPUTS=") +
        std::to_string(static_cast<long long>(problem.GetOutChannels())) +
        std::string(" -DMLO_N_INPUTS=") +
        std::to_string(static_cast<long long>(problem.GetInChannels())) +
        std::string(" -DMLO_BATCH_SZ=") +
        std::to_string(static_cast<long long>(problem.GetBatchSize())) +
        std::string(" -DMLO_OUT_WIDTH=") +
        std::to_string(static_cast<long long>(problem.GetOutWidth())) +
        std::string(" -DMLO_OUT_HEIGHT=") +
        std::to_string(static_cast<long long>(problem.GetOutHeight())) +
        std::string(" -DMLO_OUT_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetOutBatchStride())) +
        std::string(" -DMLO_OUT_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetOutChannelStride())) +
        std::string(" -DMLO_OUT_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetOutStrideH())) +
        std::string(" -DMLO_IN_WIDTH=") +
        std::to_string(static_cast<long long>(problem.GetInWidth())) +
        std::string(" -DMLO_IN_HEIGHT=") +
        std::to_string(static_cast<long long>(problem.GetInHeight())) +
        std::string(" -DMLO_IN_BATCH_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetInBatchStride())) +
        std::string(" -DMLO_IN_CHANNEL_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetInChannelStride())) +
        std::string(" -DMLO_IN_STRIDE=") +
        std::to_string(static_cast<long long>(problem.GetInStrideH()))
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
        std::to_string(static_cast<long long>(alu_tile1)) + ctx.general_compile_options;

    if(group_counts >= 2)
    {
        kernel_params.comp_options += (std::string(" -DMLO_GROUP_COUNTS=") +
                                       std::to_string(static_cast<long long>(group_counts)));
        kernel_params.comp_options +=
            (std::string(" -DMLO_GROUP_TILES=") +
             std::to_string(static_cast<long long>(problem.GetOutChannels() / group_counts)));
        kernel_params.comp_options +=
            (std::string(" -DMLO_STACK_PERGROUP=") +
             std::to_string(static_cast<long long>(
                 (problem.GetOutChannels() / group_counts + n_out_tiles_perstack - 1) /
                 n_out_tiles_perstack)));
        kernel_params.comp_options += std::string(" -DGRP_MOD_ENABLE");
    }

    kernel_params.l_wk.push_back(static_cast<size_t>(result.grp_tile1) * result.grp_tile0);
    kernel_params.l_wk.push_back(1);
    kernel_params.l_wk.push_back(1);

    size_t gbl_wk0 = static_cast<size_t>(n_out_tile_blocks0) * n_out_tile_blocks1;

    if(n_out_tiles_perstack == 0 /* DIV/0 */)
    {
        MIOPEN_LOG_E("n_out_tiles_perstack == 0");
        return {miopenStatusInternalError};
    }
    size_t gbl_wk1 =
        group_counts >= 2
            ? (((problem.GetOutChannels() / group_counts + n_out_tiles_perstack - 1) /
                n_out_tiles_perstack) *
               group_counts)
            : ((problem.GetOutChannels() + n_out_tiles_perstack - 1) / n_out_tiles_perstack);
    size_t gbl_wk2 = (problem.GetBatchSize() + result.n_stacks - 1) / result.n_stacks;

    kernel_params.g_wk.push_back(gbl_wk0 * kernel_params.l_wk[0]);
    kernel_params.g_wk.push_back(gbl_wk1);
    kernel_params.g_wk.push_back(gbl_wk2);

    kernel_params.kernel_file = "MIOpenConvDirUni.cl";
    kernel_params.kernel_name = "MIOpenConvUni";

    result.construction_params.push_back(kernel_params);
    result.invoker_factory = &miopen::conv::MakeGenericXWYPadInvoker;

    if(problem.IsDirectionForward())
        result.invoker_factory = &miopen::conv::MakeGenericXWYPadInvoker;

    return result;
}

ConvSolution ConvOclDirectFwd::GetSolution(const ExecutionContext& ctx,
                                           const ProblemDescription& problem,
                                           const LegacyPerformanceConfig& config) const
{
    ConvSolution result = BaseGetSolution(ctx, problem, config);

    if(result.Succeeded())
    {
        result.construction_params[0].comp_options +=
            std::string(" -DMLO_CONV_BIAS=") +
            std::to_string(static_cast<long long>(problem.GetBias()));
    }

    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
