/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/solver.hpp>

#include <miopen/conv/invokers/impl_gemm.hpp>
#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/implicitgemm_params.hpp>
#include <miopen/hip_build_utils.hpp>

#include "implicitgemm_util.hpp"

#define WORKAROUND_ISSUE_2174_2222_2224_2243 1

namespace miopen {
namespace solver {

bool ConvHipImplicitGemmV4_1x1::IsApplicable(const ConvolutionContext& ctx) const
{
#if WORKAROUND_SWDEV_229277_227616_229195
    if(!IsHccCompiler())
        return false;
#endif
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.Is2d())
        return false;

    return ctx.IsFp32() && ctx.pad_h == 0 && ctx.pad_w == 0 && ctx.group_counts == 1 &&
           ctx.batch_sz % 8 == 0 &&
           (ctx.batch_sz * KernelOutputHeightHo(ctx) * KernelOutputWidthWo(ctx)) % 64 == 0 &&
           ctx.n_outputs % 16 == 0 && ctx.kernel_size_h == 1 && ctx.kernel_size_w == 1 &&
           ctx.n_inputs % 8 == 0 && ctx.kernel_dilation_h == 1 && ctx.kernel_dilation_w == 1;
}

bool ConvHipImplicitGemmV4Fwd::IsApplicable(const ConvolutionContext& ctx) const
{
#if WORKAROUND_SWDEV_229277_227616_229195
    if(!IsHccCompiler())
        return false;
#endif

    if(!ctx.direction.IsForward())
        return false;

    if(!ctx.use_hip_kernels)
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32() && !ctx.IsFp16() && !ctx.IsBfp16())
        return false;

    if(ctx.group_counts != 1)
        return false;

#if WORKAROUND_ISSUE_2174_2222_2224_2243
    if(miopen::HipCompilerVersion() >= external_tool_version_t{2, 6, 0})
    {
        if(!(ctx.kernel_dilation_h == 1 && ctx.kernel_dilation_w == 1))
            return false;
    }
#endif

    // channels is divided by epack to pack 2/4 fp16/bfp16
    if(ctx.n_inputs % GetEPackLength(ctx, false) != 0)
        return false;

    // For fp16, when E=c*x*y % 32 == 0, 4 channels are accumulated through dot4 (2 * dot2)
    // operation
    // For bfp16/fp16, when E=c*x*y % 16 == 0, 2 channels are accumulated through dot2 operation
    // For fp32, when E=c*x*y % 8 == 0, no dot2 operation exist.
    const int MultipleOf = (ctx.IsFp16() || ctx.IsBfp16()) ? 16 : 8;
    if((ctx.n_inputs * ctx.kernel_size_h * ctx.kernel_size_w) % MultipleOf != 0)
        return false;

    // padding support required for out_of_bound configs
    bool no_out_of_bound = (ctx.in_width >= ((ctx.kernel_size_w - 1) * ctx.kernel_dilation_w + 1) +
                                                (ctx.out_width - 1) * ctx.kernel_stride_w) &&
                           (ctx.in_height >= ((ctx.kernel_size_h - 1) * ctx.kernel_dilation_h + 1) +
                                                 (ctx.out_height - 1) * ctx.kernel_stride_h);

    return ctx.pad_h == 0 && ctx.pad_w == 0 && ctx.batch_sz % 8 == 0 &&
           (ctx.batch_sz * ctx.out_height * ctx.out_width) % 64 == 0 && ctx.n_outputs % 16 == 0 &&
           no_out_of_bound;
}

bool ConvHipImplicitGemmV4WrW::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!ctx.direction.IsBackwardWrW())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.use_hip_kernels)
        return false;

    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;

    if(ctx.group_counts != 1)
        return false;

#if WORKAROUND_ISSUE_2174_2222_2224_2243
    if(miopen::HipCompilerVersion() >= external_tool_version_t{2, 6, 0})
    {
        if(!(ctx.kernel_stride_w == 1 && ctx.kernel_stride_h == 1))
            return false;
    }
#endif

    // batch is divided by epack to pack 2/4 fp16/bfp16
    if(ctx.batch_sz % GetEPackLength(ctx, false) != 0)
        return false;

    // For fp16, when E=c*x*y % 32 == 0, 4 channels are accumulated through dot4 (2 * dot2)
    // operation
    // For bfp16/fp16, when E=c*x*y % 16 == 0, 2 channels are accumulated through dot2 operation
    // For fp32, when E=c*x*y % 8 == 0, no dot2 operation exist.
    const int MultipleOf = (ctx.IsFp16() || ctx.IsBfp16()) ? 16 : 8;
    if((ctx.batch_sz * ctx.in_height * ctx.in_width) % MultipleOf != 0)
        return false;

    return ctx.pad_h == 0 && ctx.pad_w == 0 && ctx.n_outputs % 8 == 0 &&
           (ctx.n_outputs * ctx.kernel_size_h * ctx.kernel_size_w) % 64 == 0 &&
           ctx.n_inputs % 16 == 0;
}

PerformanceImplicitGemm
ConvHipImplicitGemmV4Fwd::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemm>(ctx);
}

PerformanceImplicitGemm
ConvHipImplicitGemmV4WrW::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemm>(ctx);
}

PerformanceImplicitGemm
ConvHipImplicitGemmV4_1x1::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemm>(ctx);
}

bool ConvHipImplicitGemmV4Fwd::IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                                        const PerformanceImplicitGemm& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4WrW::IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                                        const PerformanceImplicitGemm& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4_1x1::IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                                         const PerformanceImplicitGemm& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

using ImplicitGemmKernel_t = enum {
    ImplicitGemmV4     = 0,
    ImplicitGemmV4_1x1 = 1,
};

static inline ConvSolution GetSolutionBase(const ConvolutionContext& ctx,
                                           const PerformanceImplicitGemm& config,
                                           const ImplicitGemmKernel_t kernel,
                                           const int n,
                                           const int k,
                                           const int ho,
                                           const int wo)
{
    ConvSolution result;
    KernelInfo construction_parameters;

    const int N1 = config.GemmNRepeat;
    const int N2 = config.GemmNPerThreadSubC;

    std::size_t b = (static_cast<std::size_t>(n) * ho * wo) / (static_cast<std::size_t>(N1) * N2);

    std::size_t BPerBlock = config.BPerBlock;
    std::size_t KPerBlock = config.KPerBlock;
    std::size_t EPerBlock = config.EPerBlock;

    const int ThreadPerLevel1Cluster = config.GemmMLevel0Cluster * config.GemmNLevel0Cluster *
                                       config.GemmMLevel1Cluster * config.GemmNLevel1Cluster;

    const int block_size = ThreadPerLevel1Cluster;

    std::size_t grid_size = (b / BPerBlock) * (k / KPerBlock);

    std::size_t lkl_wk0 = block_size;
    std::size_t lkl_wk1 = 1;
    std::size_t lkl_wk2 = 1;

    construction_parameters.l_wk.push_back(lkl_wk0);
    construction_parameters.l_wk.push_back(lkl_wk1);
    construction_parameters.l_wk.push_back(lkl_wk2);

    std::size_t gbl_wk0 = lkl_wk0 * grid_size;
    std::size_t gbl_wk1 = 1;
    std::size_t gbl_wk2 = 1;

    construction_parameters.g_wk.push_back(gbl_wk0);
    construction_parameters.g_wk.push_back(gbl_wk1);
    construction_parameters.g_wk.push_back(gbl_wk2);

    if(kernel == ImplicitGemmV4)
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_implicit_gemm_v4_nchw_kcyx_nkhw_lds_double_buffer";
    }
    else if(kernel == ImplicitGemmV4_1x1)
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer";
    }
    else
    {
        MIOPEN_THROW("invalid value of 'kernel'");
    }

    std::size_t WeiBlockCopySubLengths_E = EPerBlock / config.WeiBlockCopyClusterLengths_E;
    std::size_t WeiBlockCopySubLengths_K = KPerBlock / config.WeiBlockCopyClusterLengths_K;

    std::size_t InBlockCopySubLengths_B  = BPerBlock / config.InBlockCopyClusterLengths_B;
    std::size_t InBlockCopySubLengths_N2 = N2 / config.InBlockCopyClusterLengths_N2;

    auto WeiBlockCopySrcDataPerRead_E = GetReadWriteVectorSize(WeiBlockCopySubLengths_E);
    auto InBlockCopySrcDataPerRead_B  = GetReadWriteVectorSize(InBlockCopySubLengths_B);

    InBlockCopySrcDataPerRead_B =
        ctx.kernel_size_w > 1
            ? std::min(InBlockCopySrcDataPerRead_B, GetReadWriteVectorSize(ctx.kernel_dilation_w))
            : InBlockCopySrcDataPerRead_B;

    InBlockCopySrcDataPerRead_B = ctx.kernel_stride_w > 1 ? 1 : InBlockCopySrcDataPerRead_B;

    WeiBlockCopySrcDataPerRead_E =
        ctx.direction.IsBackwardData() ? 1 : WeiBlockCopySrcDataPerRead_E;

    const auto WeiBlockCopyDstDataPerWrite_K =
        ctx.IsFp32() ? GetReadWriteVectorSize(WeiBlockCopySubLengths_K) : 1;
    const auto InBlockCopyDstDataPerWrite_N2 =
        ctx.IsFp32() ? GetReadWriteVectorSize(InBlockCopySubLengths_N2) : 1;
    const auto WeiBlockCopyDstDataPerWrite_EPack = !ctx.IsFp32() ? GetEPackLength(ctx, false) : 1;
    const auto InBlockCopyDstDataPerWrite_EPack  = !ctx.IsFp32() ? GetEPackLength(ctx, false) : 1;

    if(ctx.direction.IsBackwardWrW())
    {
        // clang-format off
        construction_parameters.comp_options =
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ctx.n_inputs) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ctx.n_outputs) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ctx.out_height) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ctx.out_width) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ctx.in_height) +  // swapped
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.in_width);
        // clang-format on
    }
    else
    {
        // clang-format off
        construction_parameters.comp_options =
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ctx.n_outputs) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ctx.n_inputs) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ctx.in_height) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ctx.in_width) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ctx.out_height) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.out_width);
        // clang-format on
    }

    if(ctx.IsFp16() || ctx.IsBfp16())
    {
        construction_parameters.comp_options +=
            std::string(" -DCK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_EPACK=") +
            std::to_string(InBlockCopyDstDataPerWrite_EPack) +
            std::string(" -DCK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_EPACK=") +
            std::to_string(WeiBlockCopyDstDataPerWrite_EPack);
    }
    else
    {

        construction_parameters.comp_options +=
            std::string(" -DCK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_N2=") +
            std::to_string(InBlockCopyDstDataPerWrite_N2) +
            std::string(" -DCK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K=") +
            std::to_string(WeiBlockCopyDstDataPerWrite_K);
    }

    // clang-format off
    construction_parameters.comp_options +=
        std::string(" -std=c++14") +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=") + std::to_string(static_cast<std::size_t>(ctx.direction.IsForward())) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=") + std::to_string(static_cast<std::size_t>(ctx.direction.IsBackwardData())) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=") + std::to_string(static_cast<std::size_t>(ctx.direction.IsBackwardWrW())) +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(ctx.batch_sz) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(ctx.kernel_size_h) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(ctx.kernel_size_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(ctx.kernel_stride_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(ctx.kernel_stride_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(ctx.kernel_dilation_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(ctx.kernel_dilation_w) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_B_PER_BLOCK=") + std::to_string(BPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_K_PER_BLOCK=") + std::to_string(KPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_E_PER_BLOCK=") + std::to_string(EPerBlock) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_PARAM_GEMM_N_REPEAT=") + std::to_string(config.GemmNRepeat) +
        std::string(" -DCK_PARAM_GEMM_M_PER_THREAD_SUB_C=") + std::to_string(config.GemmMPerThreadSubC) +
        std::string(" -DCK_PARAM_GEMM_N_PER_THREAD_SUB_C=") + std::to_string(config.GemmNPerThreadSubC) +
        std::string(" -DCK_PARAM_GEMM_M_LEVEL0_CLUSTER=") + std::to_string(config.GemmMLevel0Cluster) +
        std::string(" -DCK_PARAM_GEMM_N_LEVEL0_CLUSTER=") + std::to_string(config.GemmNLevel0Cluster) +
        std::string(" -DCK_PARAM_GEMM_M_LEVEL1_CLUSTER=") + std::to_string(config.GemmMLevel1Cluster) +
        std::string(" -DCK_PARAM_GEMM_N_LEVEL1_CLUSTER=") + std::to_string(config.GemmNLevel1Cluster) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.InBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_N1=") + std::to_string(config.InBlockCopyClusterLengths_N1) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B=") + std::to_string(config.InBlockCopyClusterLengths_B) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_N2=") + std::to_string(config.InBlockCopyClusterLengths_N2) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_SRC_DATA_PER_READ_B=") + std::to_string(InBlockCopySrcDataPerRead_B) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.WeiBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K=") + std::to_string(config.WeiBlockCopyClusterLengths_K) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E=") + std::to_string(WeiBlockCopySrcDataPerRead_E) +
        std::string(" -DCK_PARAM_EPACK_LENGTH=") + std::to_string(GetEPackLength(ctx, false)) +
        std::string(" -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx) ? '1' : '0') +
        ctx.general_compile_options;
    // clang-format on

    if(ctx.direction.IsForward() || ctx.direction.IsBackwardData())
        result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);

    result.construction_params.push_back(construction_parameters);
    return result;
}

ConvSolution ConvHipImplicitGemmV4Fwd::GetSolution(const ConvolutionContext& ctx,
                                                   const PerformanceImplicitGemm& config,
                                                   const bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmV4,
                           KernelBatchN(ctx),
                           KernelOutputChannelK(ctx),
                           KernelOutputHeightHo(ctx),
                           KernelOutputWidthWo(ctx));
}
ConvSolution ConvHipImplicitGemmV4WrW::GetSolution(const ConvolutionContext& ctx,
                                                   const PerformanceImplicitGemm& config,
                                                   const bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmV4,
                           KernelBatchN(ctx),
                           KernelOutputChannelK(ctx),
                           KernelOutputHeightHo(ctx),
                           KernelOutputWidthWo(ctx));
}

ConvSolution ConvHipImplicitGemmV4_1x1::GetSolution(const ConvolutionContext& ctx,
                                                    const PerformanceImplicitGemm& config,
                                                    const bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmV4_1x1,
                           KernelBatchN(ctx),
                           KernelOutputChannelK(ctx),
                           KernelOutputHeightHo(ctx),
                           KernelOutputWidthWo(ctx));
}

int ConvHipImplicitGemmV4Fwd::RunAndMeasureSolution(const miopen::Handle& profile_h,
                                                    ConstData_t bot_buf,
                                                    Data_t top_buf,
                                                    ConstData_t wei_buf,
                                                    ConstData_t bias_buf,
                                                    const ConvolutionContext& ctx,
                                                    const ConvSolution& solution,
                                                    float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;

    return RunAndMeasureSolutionBase(
        profile_h, bot_buf, top_buf, wei_buf, ctx, solution, elapsed_time);
}

int ConvHipImplicitGemmV4WrW::RunAndMeasureSolution(const miopen::Handle& profile_h,
                                                    ConstData_t bot_buf,
                                                    ConstData_t top_buf,
                                                    Data_t wei_buf,
                                                    ConstData_t bias_buf,
                                                    const ConvolutionContext& ctx,
                                                    const ConvSolution& solution,
                                                    float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;
    return RunAndMeasureSolutionBase(
        profile_h, bot_buf, top_buf, wei_buf, ctx, solution, elapsed_time);
}

int ConvHipImplicitGemmV4_1x1::RunAndMeasureSolution(const miopen::Handle& profile_h,
                                                     ConstData_t bot_buf,
                                                     Data_t top_buf,
                                                     ConstData_t wei_buf,
                                                     ConstData_t bias_buf,
                                                     const ConvolutionContext& ctx,
                                                     const ConvSolution& solution,
                                                     float& elapsed_time) const
{
    assert(bias_buf == nullptr);
    (void)bias_buf;
    return RunAndMeasureSolutionBase(
        profile_h, bot_buf, top_buf, wei_buf, ctx, solution, elapsed_time);
}

PerformanceImplicitGemm ConvHipImplicitGemmV4Fwd::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}
PerformanceImplicitGemm ConvHipImplicitGemmV4WrW::Search(const ConvolutionContext& context) const
{
    return GenericSearchWrW(*this, context);
}

PerformanceImplicitGemm ConvHipImplicitGemmV4_1x1::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}

} // namespace solver
} // namespace miopen
