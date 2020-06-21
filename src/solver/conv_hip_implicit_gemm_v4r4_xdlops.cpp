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
#include <miopen/env.hpp>

#include "implicitgemm_util.hpp"

namespace miopen {
namespace solver {

// fail with vector load for some cases
/// \todo enable vector load after fix it
#define WORKAROUND_FAILED_VECTOR_LOAD 1

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4FwdXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

static inline ConvSolution GetSolutionBase(const ConvolutionContext& ctx,
                                           const PerformanceImplicitGemmXdlops& config,
                                           const ImplicitGemmXdlopsKernel kernel,
                                           const int n,
                                           const int k,
                                           const int ho,
                                           const int wo)
{
    ConvSolution result;
    KernelInfo construction_parameters;

    std::size_t b = (n * ho * wo);

    std::size_t BPerBlock = config.BPerBlock;
    std::size_t KPerBlock = config.KPerBlock;
    std::size_t EPerBlock = config.EPerBlock;

    const int WaveSize = 64;
    std::size_t block_size =
        BPerBlock * KPerBlock / (config.GemmMPerWave * config.GemmNPerWave) * WaveSize;

    std::size_t grid_size = (b / BPerBlock) * (k / KPerBlock);

    const int Y = KernelFilterHeightY(ctx);
    const int X = KernelFilterWidthX(ctx);
    const int C = KernelInputChannelC(ctx);
    const int E = C * Y * X;

    std::size_t global_load_size =
        (BPerBlock + KPerBlock) * E * grid_size * GetTypeSize(ctx.in_data_type);
    std::size_t global_store_size = k * b * GetTypeSize(ctx.in_data_type);
    std::size_t global_size       = global_load_size + global_store_size;

    MIOPEN_LOG_I2("global load/store size = "
                  << static_cast<float>(global_size) / 1024 / 1024 / 1024
                  << " GB.");

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

    if(kernel == ImplicitGemmXdlopsKernel::KernelFwdWrw)
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_lds_double_buffer.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw_lds_double_buffer";
    }
    else
    {
        construction_parameters.kernel_file =
            "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer.cpp";

        construction_parameters.kernel_name =
            "gridwise_convolution_implicit_gemm_v4r4_xdlops_nchw_kc1x1_nkhw_lds_double_buffer";
    }

    std::size_t WeiBlockCopySubLengths_E = EPerBlock / config.WeiBlockCopyClusterLengths_E;
    std::size_t WeiBlockCopySubLengths_K = KPerBlock / config.WeiBlockCopyClusterLengths_K;

    auto WeiBlockCopySrcDataPerRead_E  = GetReadWriteVectorSize(WeiBlockCopySubLengths_E);
    auto WeiBlockCopyDstDataPerWrite_K = GetReadWriteVectorSize(WeiBlockCopySubLengths_K);

    int OutThreadCopyDataPerAccess_B = 1;

#if WORKAROUND_FAILED_VECTOR_LOAD
    int InBlockCopyDataPerAccess_B = 1;
#else
    std::size_t InBlockCopySubLengths_B = BPerBlock / config.InBlockCopyClusterLengths_B;
    auto InBlockCopyDataPerAccess_B     = GetReadWriteVectorSize(InBlockCopySubLengths_B);
#endif

    WeiBlockCopySrcDataPerRead_E =
        ctx.direction.IsBackwardData() ? 1 : WeiBlockCopySrcDataPerRead_E;

    if(ctx.IsFp16() || ctx.IsBfp16())
    {
        WeiBlockCopySrcDataPerRead_E  = 1;
        WeiBlockCopyDstDataPerWrite_K = 1;
    }

    const ImplicitGemmDirection direction =
        ctx.direction.IsForward()
            ? ImplicitGemmDirection::ForwardData
            : (ctx.direction.IsBackwardData() ? ImplicitGemmDirection::BackwardData
                                              : ImplicitGemmDirection::BackwardWeight);
    if(ctx.direction.IsBackwardWrW())
        WeiBlockCopySrcDataPerRead_E =
            (X * Y) % WeiBlockCopySubLengths_E != 0 ? 1 : WeiBlockCopySrcDataPerRead_E;

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

    // clang-format off
    construction_parameters.comp_options +=
        std::string(" -std=c++14 ") +
        std::string(" -DCK_PARAM_PROBLEM_DIRECTION=") + std::to_string(static_cast<int>(direction)) +
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
        std::string(" -DCK_PARAM_GEMM_M_PER_WAVE=") + std::to_string(config.GemmMPerWave) +
        std::string(" -DCK_PARAM_GEMM_N_PER_WAVE=") + std::to_string(config.GemmNPerWave) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.InBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_CLUSTER_LENGTHS_B=") + std::to_string(config.InBlockCopyClusterLengths_B) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.WeiBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K=") + std::to_string(config.WeiBlockCopyClusterLengths_K) +
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_DATA_PER_ACCESS_B=") + std::to_string(InBlockCopyDataPerAccess_B) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E=") + std::to_string(WeiBlockCopySrcDataPerRead_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K=") + std::to_string(WeiBlockCopyDstDataPerWrite_K) +
        std::string(" -DCK_PARAM_OUT_THREAD_COPY_DATA_PER_ACCESS_B=") + std::to_string(OutThreadCopyDataPerAccess_B) +
        std::string(" -DCK_PARAM_EPACK_LENGTH=") + std::to_string(GetEPackLength(ctx, true)) +
        std::string(" -DCK_USE_AMD_XDLOPS=") + (IsXdlopsSupport(ctx) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + (miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM{}) ? '1' : '0') +
        std::string(" -DCK_USE_AMD_XDLOPS_EMULATE=") + (miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}) ? '1' : '0') +
        ctx.general_compile_options;
    // clang-format on

    if(ctx.direction.IsForward() || ctx.direction.IsBackwardData())
        result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);

    result.construction_params.push_back(construction_parameters);
    return result;
}

ConvSolution ConvHipImplicitGemmV4R4FwdXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmXdlopsKernel::KernelFwdWrw,
                           KernelBatchN(ctx),
                           KernelOutputChannelK(ctx),
                           KernelOutputHeightHo(ctx),
                           KernelOutputWidthWo(ctx));
}

ConvSolution ConvHipImplicitGemmV4R4Xdlops_1x1::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmXdlopsKernel::Kernel1x1,
                           KernelBatchN(ctx),
                           KernelOutputChannelK(ctx),
                           KernelOutputHeightHo(ctx),
                           KernelOutputWidthWo(ctx));
}

ConvSolution ConvHipImplicitGemmV4R4WrWXdlops::GetSolution(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& config, bool) const
{
    return GetSolutionBase(ctx,
                           config,
                           ImplicitGemmXdlopsKernel::KernelFwdWrw,
                           KernelBatchN(ctx),
                           KernelOutputChannelK(ctx),
                           KernelOutputHeightHo(ctx),
                           KernelOutputWidthWo(ctx));
}

int ConvHipImplicitGemmV4R4FwdXdlops::RunAndMeasureSolution(const miopen::Handle& profile_h,
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

int ConvHipImplicitGemmV4R4Xdlops_1x1::RunAndMeasureSolution(const miopen::Handle& profile_h,
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

int ConvHipImplicitGemmV4R4WrWXdlops::RunAndMeasureSolution(const miopen::Handle& profile_h,
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

bool ConvHipImplicitGemmV4R4FwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.direction.IsForward())
        return false;
    if(!ctx.Is2d())
        return false;

    // padding support required for out_of_bound configs
    bool no_out_of_bound = (ctx.in_width >= ((ctx.kernel_size_w - 1) * ctx.kernel_dilation_w + 1) +
                                                (ctx.out_width - 1) * ctx.kernel_stride_w) &&
                           (ctx.in_height >= ((ctx.kernel_size_h - 1) * ctx.kernel_dilation_h + 1) +
                                                 (ctx.out_height - 1) * ctx.kernel_stride_h);

    return IsApplicableXdlops(ctx) && no_out_of_bound && ctx.pad_h == 0 && ctx.pad_w == 0 &&
           ctx.group_counts == 1;
}

bool ConvHipImplicitGemmV4R4Xdlops_1x1::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!ctx.use_hip_kernels)
        return false;

    return IsApplicableXdlops(ctx) && ctx.Is2d() && ctx.IsFp32() && ctx.pad_h == 0 &&
           ctx.pad_w == 0 && ctx.group_counts == 1 && ctx.kernel_size_h == 1 &&
           ctx.kernel_size_w == 1;
}

bool ConvHipImplicitGemmV4R4WrWXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!ctx.direction.IsBackwardWrW())
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!ctx.Is2d())
        return false;
    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;

    return IsApplicableXdlops(ctx) && ctx.pad_h == 0 && ctx.pad_w == 0 && ctx.group_counts == 1;
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4Xdlops_1x1::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4WrWXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

bool ConvHipImplicitGemmV4R4FwdXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4R4Xdlops_1x1::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4R4WrWXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4Xdlops_1x1::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchFwd(*this, ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4FwdXdlops::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchFwd(*this, ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4WrWXdlops::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchWrW(*this, ctx);
}

} // namespace solver
} // namespace miopen
