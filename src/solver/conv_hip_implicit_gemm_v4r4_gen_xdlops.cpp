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

#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include <miopen/generic_search.hpp>
#include "miopen/stringutils.hpp"
#include "implicitgemm_util.hpp"
#include "miopen/implicitgemm_params.hpp"
#include <miopen/env.hpp>

namespace miopen {
namespace solver {

// fail with vector load for some cases
/// \todo enable vector load after fix it
#define WORKAROUND_FAILED_VECTOR_LOAD 1

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4GenFwdXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
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

    // In bwd wrw, kernel's output height ho = y(filter's height) and output width wo=x(filter's
    // width)
    std::size_t b = (n * ho * wo);

    std::size_t GemmNPerBlock = config.BPerBlock;
    std::size_t GemmMPerBlock = config.KPerBlock;
    std::size_t GemmKPerBlock = config.EPerBlock;

    std::size_t block_size =
        GemmNPerBlock * GemmMPerBlock / (config.GemmMPerWave * config.GemmNPerWave) * wave_size;
    std::size_t grid_size = (b / GemmNPerBlock) * (k / GemmMPerBlock);

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
        if(ctx.group_counts > 1)
        {

            construction_parameters.kernel_file =
                "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_"
                "gnchw_gkcyx_gnkhw_lds_double_buffer.cpp";

            construction_parameters.kernel_name = "gridwise_convolution_implicit_gemm_v4r4_gen_"
                                                  "xdlops_gnchw_gkcyx_gnkhw_lds_double_buffer";
        }
        else
        {

            construction_parameters.kernel_file =
                "gridwise_convolution_implicit_gemm_v4r4_gen_xdlops_"
                "nchw_kcyx_nkhw_lds_double_buffer.cpp";

            construction_parameters.kernel_name = "gridwise_convolution_implicit_gemm_v4r4_gen_"
                                                  "xdlops_nchw_kcyx_nkhw_lds_double_buffer";
        }
    }

    std::size_t ABlockCopySubLengths_GemmK = GemmKPerBlock / config.WeiBlockCopyClusterLengths_E;
    std::size_t ABlockCopySubLengths_GemmM = GemmMPerBlock / config.WeiBlockCopyClusterLengths_K;
    std::size_t BBlockCopySubLengths_GemmN = GemmNPerBlock / config.InBlockCopyClusterLengths_B;

// Ensure that vectorized b is read via right alignment from global memory
// Consider slicing window's stride and dilation to ensure global memory reads of B are aligned
// with vector length.
#if WORKAROUND_FAILED_VECTOR_LOAD
    std::size_t BBlockCopySrcDataPerRead_B = 1;
#else
    size_t kernel_filter_x          = KernelFilterWidthX(ctx);
    size_t kernel_filter_stride_w   = KernelFilterStrideW(ctx);
    size_t kernel_filter_dilation_w = KernelFilterDilationW(ctx);

    auto BBlockCopySrcDataPerRead_B = GetReadWriteVectorSize(BBlockCopySubLengths_GemmN);
    BBlockCopySrcDataPerRead_B =
        kernel_filter_x > 1
            ? std::min(BBlockCopySrcDataPerRead_B, GetReadWriteVectorSize(kernel_filter_dilation_w))
            : BBlockCopySrcDataPerRead_B;
    BBlockCopySrcDataPerRead_B = kernel_filter_stride_w > 1 ? 1 : BBlockCopySrcDataPerRead_B;
#endif

    const int Y = KernelFilterHeightY(ctx);
    const int X = KernelFilterWidthX(ctx);

    // Disable vectorized read in backward data case. Why?
    unsigned int ABlockCopySrcDataPerRead_E = 1;
    if(ctx.IsFp32())
    {
        ABlockCopySrcDataPerRead_E = GetReadWriteVectorSize(ABlockCopySubLengths_GemmK);
    }
    else
    {
        // For fp32, E = C*Y*X
        // For fp16, E = C/EPack * Y * X
        // Since C/EPack are not in contiguous memory along with Y*X, vector length
        // can' be more than Y*X
        ABlockCopySrcDataPerRead_E = (X * Y) % ABlockCopySubLengths_GemmK != 0
                                         ? 1
                                         : GetReadWriteVectorSize(ABlockCopySubLengths_GemmK);
    }

    if(ctx.direction.IsBackwardWrW())
        ABlockCopySrcDataPerRead_E =
            (X * Y) % ABlockCopySubLengths_GemmK != 0 ? 1 : ABlockCopySrcDataPerRead_E;

    ABlockCopySrcDataPerRead_E = ctx.direction.IsBackwardData() ? 1 : ABlockCopySrcDataPerRead_E;

    std::size_t BBlockCopyDstDataPerWrite_GemmN     = 1;
    std::size_t BBlockCopyDstDataPerWrite_GemmKPACK = 1;
    std::size_t ABlockCopyDstDataPerWrite_GemmM     = 1;
    std::size_t ABlockCopyDstDataPerWrite_GemmKPACK = 1;

    if(ctx.IsFp32())
    {
        BBlockCopyDstDataPerWrite_GemmN = GetReadWriteVectorSize(BBlockCopySubLengths_GemmN);
        (void)BBlockCopyDstDataPerWrite_GemmKPACK;
        ABlockCopyDstDataPerWrite_GemmM = GetReadWriteVectorSize(ABlockCopySubLengths_GemmM);
        (void)ABlockCopyDstDataPerWrite_GemmKPACK;
    }
    else
    {
        BBlockCopyDstDataPerWrite_GemmKPACK = GetEPackLength(ctx, true);
        (void)BBlockCopyDstDataPerWrite_GemmN;
        ABlockCopyDstDataPerWrite_GemmKPACK = GetEPackLength(ctx, true);
        (void)ABlockCopyDstDataPerWrite_GemmM;
    }

    const ImplicitGemmDirection direction =
        ctx.direction.IsForward()
            ? ImplicitGemmDirection::ForwardData
            : (ctx.direction.IsBackwardData() ? ImplicitGemmDirection::BackwardData
                                              : ImplicitGemmDirection::BackwardWeight);

    if(ctx.direction.IsBackwardWrW())
    {
        // clang-format off
        construction_parameters.comp_options =
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(ctx.n_inputs) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(ctx.n_outputs) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(ctx.out_height) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(ctx.out_width) + // swapped
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ctx.in_height) +  // swapped
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.in_width) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=") + std::to_string(1);
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
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(ctx.out_width) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=") + std::to_string(1) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=") + std::to_string(0);
        // clang-format on
    }

    // Compute correct padding values
    std::size_t in_height  = ctx.in_height;
    std::size_t in_width   = ctx.in_width;
    std::size_t out_height = ctx.out_height;
    std::size_t out_width  = ctx.out_width;

    if(ctx.direction.IsBackwardWrW())
    {
        in_height  = ctx.out_height; // unswap
        in_width   = ctx.out_width;  // unswap
        out_height = ctx.in_height;  // unswap
        out_width  = ctx.in_width;   // unswap
    }

    // adjust padding size to align with the way MIOpen deal with padding
    std::size_t left_pad_h = ctx.pad_h;
    std::size_t left_pad_w = ctx.pad_w;

    std::size_t hi_padded = 1 + (ctx.kernel_size_h - 1) * ctx.kernel_dilation_h +
                            (out_height - 1) * ctx.kernel_stride_h;
    std::size_t wi_padded =
        1 + (ctx.kernel_size_w - 1) * ctx.kernel_dilation_w + (out_width - 1) * ctx.kernel_stride_w;

    std::size_t right_pad_h =
        hi_padded > (left_pad_h + in_height) ? hi_padded - (left_pad_h + in_height) : 0;
    std::size_t right_pad_w =
        wi_padded > (left_pad_w + in_width) ? wi_padded - (left_pad_w + in_width) : 0;

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
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_H=") + std::to_string(left_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_W=") + std::to_string(left_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_H=") + std::to_string(right_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_W=") + std::to_string(right_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_GROUP_COUNTS=") + std::to_string(ctx.group_counts) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=") + std::to_string(GemmNPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=") + std::to_string(GemmMPerBlock) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=") + std::to_string(GemmKPerBlock) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_PARAM_GEMM_M_PER_WAVE=") + std::to_string(config.GemmMPerWave) +
        std::string(" -DCK_PARAM_GEMM_N_PER_WAVE=") + std::to_string(config.GemmNPerWave) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(config.InBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(config.InBlockCopyClusterLengths_B) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(config.WeiBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=") + std::to_string(config.WeiBlockCopyClusterLengths_K) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(BBlockCopySrcDataPerRead_B) +
        std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K=") + std::to_string(ABlockCopySrcDataPerRead_E) +
        std::string(" -DCK_PARAM_GEMM_KPACK_LENGTH=") + std::to_string(GetEPackLength(ctx,true)) +
        std::string(" -DCK_USE_AMD_XDLOPS=") + std::to_string(IsXdlopsSupport(ctx) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_INLINE_ASM=") + std::to_string(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_XDLOPS_INLINE_ASM{}) ? 1 : 0) +
        std::string(" -DCK_USE_AMD_XDLOPS_EMULATE=") + (miopen::IsEnabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_XDLOPS_EMULATE{}) ? '1' : '0') +
        std::string(" -D__HIP_PLATFORM_HCC__=1") +
        ctx.general_compile_options;
    // clang-format on

    if(ctx.IsFp32())
    {
        construction_parameters.comp_options +=
            std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N=") +
            std::to_string(BBlockCopyDstDataPerWrite_GemmN) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M=") +
            std::to_string(ABlockCopyDstDataPerWrite_GemmM);
    }
    else
    {
        construction_parameters.comp_options +=
            std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") +
            std::to_string(BBlockCopyDstDataPerWrite_GemmKPACK) +
            std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_KPACK=") +
            std::to_string(ABlockCopyDstDataPerWrite_GemmKPACK);
    }

    result.construction_params.push_back(construction_parameters);
    return result;
}

ConvSolution ConvHipImplicitGemmV4R4GenFwdXdlops::GetSolution(
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

ConvSolution ConvHipImplicitGemmV4R4GenWrWXdlops::GetSolution(
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

int ConvHipImplicitGemmV4R4GenFwdXdlops::RunAndMeasureSolution(miopen::Handle& profile_h,
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

int ConvHipImplicitGemmV4R4GenWrWXdlops::RunAndMeasureSolution(miopen::Handle& profile_h,
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

bool ConvHipImplicitGemmV4R4GenFwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!IsXdlopsSupport(ctx))
        return false;

    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;

    if(!ctx.direction.IsForward())
        return false;

    if(!ctx.Is2d())
        return false;

    std::size_t k  = ctx.n_outputs / ctx.group_counts;
    std::size_t c  = ctx.n_inputs / ctx.group_counts;
    const auto& n  = ctx.batch_sz;
    const auto& y  = ctx.kernel_size_h;
    const auto& x  = ctx.kernel_size_w;
    const auto& ho = ctx.out_height;
    const auto& wo = ctx.out_width;

    // channels is divided by epack to pack 2/4 fp16/bfp16
    if(ctx.n_inputs % GetEPackLength(ctx, true) != 0)
        return false;

    const auto nonVectorizedC = c / GetEPackLength(ctx, true);

    const auto WaveSize = 64;

    // For fp16, when c*x*y % 4 == 0, 4 channels are accumulated through dot4 (2 * dot2) operation
    const auto e = nonVectorizedC * y * x;
    const auto b = n * ho * wo;
    return (k * b) % 256 == 0 && (e * b) % WaveSize == 0 && (e * k) % WaveSize == 0 &&
           b % 16 == 0 && e % 4 == 0 && k % 4 == 0;
}

bool ConvHipImplicitGemmV4R4GenWrWXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!IsXdlopsSupport(ctx))
        return false;

    if(!(ctx.IsFp32() || ctx.IsFp16() || ctx.IsBfp16()))
        return false;

    if(!ctx.direction.IsBackwardWrW())
        return false;

    if(!ctx.Is2d())
        return false;

    const auto& n_eqv = KernelBatchN(ctx);
    const auto& k_eqv = KernelOutputChannelK(ctx);
    const auto& c_eqv = KernelInputChannelC(ctx);

    const auto& ho_eqv = KernelOutputHeightHo(ctx);
    const auto& wo_eqv = KernelOutputWidthWo(ctx);

    const auto& y_eqv = KernelFilterHeightY(ctx);
    const auto& x_eqv = KernelFilterWidthX(ctx);

    // batch is divided by epack to pack 2/4 fp16/bfp16
    if(c_eqv % GetEPackLength(ctx, true) != 0)
        return false;

    const auto nonVectorizedC = c_eqv / GetEPackLength(ctx, true);

    const auto WaveSize = 64;

    const auto b = n_eqv * ho_eqv * wo_eqv;
    const auto e = nonVectorizedC * y_eqv * x_eqv;

    return (k_eqv * b) % 256 == 0 && (e * b) % WaveSize == 0 && (e * k_eqv) % WaveSize == 0 &&
           b % 16 == 0 && e % 4 == 0 && k_eqv % 4 == 0;
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4GenWrWXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemmXdlops>(ctx);
}

bool ConvHipImplicitGemmV4R4GenFwdXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4R4GenWrWXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceImplicitGemmXdlops& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4GenFwdXdlops::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchFwd(*this, ctx);
}

PerformanceImplicitGemmXdlops
ConvHipImplicitGemmV4R4GenWrWXdlops::Search(const ConvolutionContext& ctx) const
{
    return GenericSearchFwd(*this, ctx);
}

} // namespace solver
} // namespace miopen
