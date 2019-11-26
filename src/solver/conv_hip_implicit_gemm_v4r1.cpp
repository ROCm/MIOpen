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
#include <cstddef>
#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include <miopen/generic_search.hpp>
#include "implicitgemm_util.hpp"

namespace miopen {
namespace solver {

bool ConvHipImplicitGemmV4R1Fwd::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!ctx.direction.IsForward())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32() && !ctx.IsFp16() && !ctx.IsBfp16())
        return false;

    if(ctx.group_counts != 1)
        return false;

    std::size_t n         = ctx.batch_sz;
    std::size_t k         = ctx.n_outputs;
    std::size_t c         = ctx.n_inputs;
    std::size_t y         = ctx.kernel_size_h;
    std::size_t x         = ctx.kernel_size_w;
    std::size_t ho        = ctx.out_height;
    std::size_t wo        = ctx.out_width;
    std::size_t eMultiple = (ctx.IsFp16() || ctx.IsBfp16()) ? 16 : 8;

    // batch is divided by epack to pack 2/4 fp16/bfp16
    if(c % GetEPackLength(ctx, false) != 0)
        return false;

    return n % 8 == 0 && (n * ho * wo) % 64 == 0 && (c * y * x) % eMultiple == 0 && k % 16 == 0;
}

bool ConvHipImplicitGemmV4R1WrW::IsApplicable(const ConvolutionContext& ctx) const
{
    if(!ctx.direction.IsBackwardWrW())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32() && !ctx.IsFp16() && !ctx.IsBfp16())
        return false;

    if(ctx.group_counts != 1)
        return false;

    // retrieve dimension from ConvolutionContext
    // remember: ConvolutionContext has swapped some dimensions for you!
    // undo the swap to avoid confusion
    std::size_t n  = ctx.batch_sz;
    std::size_t k  = ctx.n_inputs;  // unswap
    std::size_t c  = ctx.n_outputs; // unswap
    std::size_t y  = ctx.kernel_size_h;
    std::size_t x  = ctx.kernel_size_w;
    std::size_t ho = ctx.in_height; // unswap
    std::size_t wo = ctx.in_width;  // unswap

    // equivalent dimension for bwd-wrw
    std::size_t n_eqv     = c;
    std::size_t k_eqv     = k;
    std::size_t c_eqv     = n;
    std::size_t y_eqv     = ho;
    std::size_t x_eqv     = wo;
    std::size_t ho_eqv    = y;
    std::size_t wo_eqv    = x;
    std::size_t eMultiple = (ctx.IsFp16() || ctx.IsBfp16()) ? 16 : 8;

    // batch is divided by epack to pack 2/4 fp16/bfp16
    if(c_eqv % GetEPackLength(ctx, false) != 0)
        return false;

    return n_eqv % 8 == 0 && (n_eqv * ho_eqv * wo_eqv) % 64 == 0 &&
           (c_eqv * y_eqv * x_eqv) % eMultiple == 0 && k_eqv % 16 == 0;
}

PerformanceImplicitGemm
ConvHipImplicitGemmV4R1Fwd::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemm>(ctx);
}

PerformanceImplicitGemm
ConvHipImplicitGemmV4R1WrW::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    return GetPerformanceConfigBase<PerformanceImplicitGemm>(ctx);
}

bool ConvHipImplicitGemmV4R1Fwd::IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                                          const PerformanceImplicitGemm& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

bool ConvHipImplicitGemmV4R1WrW::IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                                          const PerformanceImplicitGemm& c) const
{
    MIOPEN_LOG_I("");
    return c.IsValidValue() && c.IsValid(ctx);
}

int ConvHipImplicitGemmV4R1Fwd::RunAndMeasureSolution(miopen::Handle& profile_h,
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

int ConvHipImplicitGemmV4R1WrW::RunAndMeasureSolution(miopen::Handle& profile_h,
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

PerformanceImplicitGemm ConvHipImplicitGemmV4R1Fwd::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}
PerformanceImplicitGemm ConvHipImplicitGemmV4R1WrW::Search(const ConvolutionContext& context) const
{
    return GenericSearchFwd(*this, context);
}

ConvSolution ConvHipImplicitGemmV4R1Fwd::GetSolution(const ConvolutionContext& ctx,
                                                     const PerformanceImplicitGemm& config,
                                                     bool) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    const int N1 = config.GemmNRepeat;
    const int N2 = config.GemmNPerThreadSubC;

    // retrieve dimension from ConvolutionContex
    std::size_t n               = ctx.batch_sz;
    std::size_t k               = ctx.n_outputs;
    std::size_t c               = ctx.n_inputs;
    std::size_t hi              = ctx.in_height;
    std::size_t wi              = ctx.in_width;
    std::size_t ho              = ctx.out_height;
    std::size_t wo              = ctx.out_width;
    std::size_t y               = ctx.kernel_size_h;
    std::size_t x               = ctx.kernel_size_w;
    std::size_t conv_stride_h   = ctx.kernel_stride_h;
    std::size_t conv_stride_w   = ctx.kernel_stride_w;
    std::size_t conv_dilation_h = ctx.kernel_dilation_h;
    std::size_t conv_dilation_w = ctx.kernel_dilation_w;

    // adjust padding size to align with the way MIOpen deal with padding
    std::size_t left_pad_h = ctx.pad_h;
    std::size_t left_pad_w = ctx.pad_w;

    std::size_t hi_padded = 1 + (y - 1) * conv_dilation_h + (ho - 1) * conv_stride_h;
    std::size_t wi_padded = 1 + (x - 1) * conv_dilation_w + (wo - 1) * conv_stride_w;

    std::size_t right_pad_h = hi_padded > (left_pad_h + hi) ? hi_padded - (left_pad_h + hi) : 0;
    std::size_t right_pad_w = wi_padded > (left_pad_w + wi) ? wi_padded - (left_pad_w + wi) : 0;

    std::size_t b = (static_cast<std::size_t>(n) * ho * wo) / (static_cast<std::size_t>(N1) * N2);

    std::size_t b_per_block = config.BPerBlock;
    std::size_t k_per_block = config.KPerBlock;
    std::size_t e_per_block = config.EPerBlock;

    const int ThreadPerLevel1Cluster = config.GemmMLevel0Cluster * config.GemmNLevel0Cluster *
                                       config.GemmMLevel1Cluster * config.GemmNLevel1Cluster;

    const int block_size = ThreadPerLevel1Cluster;

    std::size_t grid_size = (b / b_per_block) * (k / k_per_block);

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

    construction_parameters.kernel_file =
        "gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer.cpp";

    construction_parameters.kernel_name =
        "gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer";

    std::size_t WeiBlockCopySubLengths_E = e_per_block / config.WeiBlockCopyClusterLengths_E;
    std::size_t WeiBlockCopySubLengths_K = k_per_block / config.WeiBlockCopyClusterLengths_K;

    int WeiBlockCopySrcDataPerRead_E  = GetReadWriteVectorSize(WeiBlockCopySubLengths_E);
    int WeiBlockCopyDstDataPerWrite_K = GetReadWriteVectorSize(WeiBlockCopySubLengths_K);

    std::size_t InBlockCopySubLengths_B  = b_per_block / config.InBlockCopyClusterLengths_B;
    std::size_t InBlockCopySubLengths_N2 = N2 / config.InBlockCopyClusterLengths_N2;

    int InBlockCopySrcDataPerRead_B   = GetReadWriteVectorSize(InBlockCopySubLengths_B);
    int InBlockCopyDstDataPerWrite_N2 = GetReadWriteVectorSize(InBlockCopySubLengths_N2);

    // Borrowed from non-padded version of v4
    InBlockCopySrcDataPerRead_B =
        ctx.kernel_size_w > 1
            ? std::min(InBlockCopySrcDataPerRead_B, GetReadWriteVectorSize(ctx.kernel_dilation_w))
            : InBlockCopySrcDataPerRead_B;
    InBlockCopySrcDataPerRead_B = ctx.kernel_stride_w > 1 ? 1 : InBlockCopySrcDataPerRead_B;

    // clang-format off
    construction_parameters.comp_options = 
        std::string(" -std=c++14 ") +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(n) +
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(k) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(c) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(hi) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(wi) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ho) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(wo) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(y) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(x) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(conv_stride_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(conv_stride_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(conv_dilation_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(conv_dilation_w) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_H=") + std::to_string(left_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_W=") + std::to_string(left_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_H=") + std::to_string(right_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_W=") + std::to_string(right_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=") + std::to_string(1) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=") + std::to_string(0) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_B_PER_BLOCK=") + std::to_string(b_per_block) +
        std::string(" -DCK_PARAM_TUNABLE_K_PER_BLOCK=") + std::to_string(k_per_block) +
        std::string(" -DCK_PARAM_TUNABLE_E_PER_BLOCK=") + std::to_string(e_per_block) +
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
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_N2=") + std::to_string(InBlockCopyDstDataPerWrite_N2) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.WeiBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K=") + std::to_string(config.WeiBlockCopyClusterLengths_K) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E=") + std::to_string(WeiBlockCopySrcDataPerRead_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K=") + std::to_string(WeiBlockCopyDstDataPerWrite_K) +
        std::string(" -DCK_PARAM_EPACK_LENGTH=") + std::to_string(GetEPackLength(ctx, false)) +
        std::string(" -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx) ? '1' : '0') +
        std::string(" -D__HIP_PLATFORM_HCC__=1") +
        ctx.general_compile_options;
    // clang-format on

    result.construction_params.push_back(construction_parameters);
    return result;
}

ConvSolution ConvHipImplicitGemmV4R1WrW::GetSolution(const ConvolutionContext& ctx,
                                                     const PerformanceImplicitGemm& config,
                                                     bool) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    const int N1 = config.GemmNRepeat;
    const int N2 = config.GemmNPerThreadSubC;

    // retrieve dimension from ConvolutionContex
    // remember: ConvolutionContext has swapped some dimensions for you!
    // undo the swap to avoid confusion
    std::size_t n               = ctx.batch_sz;
    std::size_t k               = ctx.n_inputs;   // unswap
    std::size_t c               = ctx.n_outputs;  // unswap
    std::size_t hi              = ctx.out_height; // unswap
    std::size_t wi              = ctx.out_width;  // unswap
    std::size_t ho              = ctx.in_height;  // unswap
    std::size_t wo              = ctx.in_width;   // unswap
    std::size_t y               = ctx.kernel_size_h;
    std::size_t x               = ctx.kernel_size_w;
    std::size_t conv_stride_h   = ctx.kernel_stride_h;
    std::size_t conv_stride_w   = ctx.kernel_stride_w;
    std::size_t conv_dilation_h = ctx.kernel_dilation_h;
    std::size_t conv_dilation_w = ctx.kernel_dilation_w;

    // adjust padding size to align with the way MIOpen deal with padding
    std::size_t left_pad_h = ctx.pad_h;
    std::size_t left_pad_w = ctx.pad_w;

    std::size_t hi_padded = 1 + (y - 1) * conv_dilation_h + (ho - 1) * conv_stride_h;
    std::size_t wi_padded = 1 + (x - 1) * conv_dilation_w + (wo - 1) * conv_stride_w;

    std::size_t right_pad_h = hi_padded > (left_pad_h + hi) ? hi_padded - (left_pad_h + hi) : 0;
    std::size_t right_pad_w = wi_padded > (left_pad_w + wi) ? wi_padded - (left_pad_w + wi) : 0;

    // equivalent dimension for bwd-wrw
    std::size_t n_eqv  = c;
    std::size_t ho_eqv = y;
    std::size_t wo_eqv = x;

    std::size_t b =
        (static_cast<std::size_t>(n_eqv) * ho_eqv * wo_eqv) / (static_cast<std::size_t>(N1) * N2);

    std::size_t b_per_block = config.BPerBlock;
    std::size_t k_per_block = config.KPerBlock;
    std::size_t e_per_block = config.EPerBlock;

    const int ThreadPerLevel1Cluster = config.GemmMLevel0Cluster * config.GemmNLevel0Cluster *
                                       config.GemmMLevel1Cluster * config.GemmNLevel1Cluster;

    const int block_size  = ThreadPerLevel1Cluster;
    std::size_t grid_size = (b / b_per_block) * (k / k_per_block);

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

    construction_parameters.kernel_file =
        "gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer.cpp";

    construction_parameters.kernel_name =
        "gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer";

    std::size_t WeiBlockCopySubLengths_E = e_per_block / config.WeiBlockCopyClusterLengths_E;
    std::size_t WeiBlockCopySubLengths_K = k_per_block / config.WeiBlockCopyClusterLengths_K;

    int WeiBlockCopySrcDataPerRead_E  = GetReadWriteVectorSize(WeiBlockCopySubLengths_E);
    int WeiBlockCopyDstDataPerWrite_K = GetReadWriteVectorSize(WeiBlockCopySubLengths_K);

    std::size_t InBlockCopySubLengths_B  = b_per_block / config.InBlockCopyClusterLengths_B;
    std::size_t InBlockCopySubLengths_N2 = N2 / config.InBlockCopyClusterLengths_N2;

    int InBlockCopySrcDataPerRead_B   = GetReadWriteVectorSize(InBlockCopySubLengths_B);
    int InBlockCopyDstDataPerWrite_N2 = GetReadWriteVectorSize(InBlockCopySubLengths_N2);

    // Borrowed from non-padded version of v4
    InBlockCopySrcDataPerRead_B =
        ctx.kernel_size_w > 1
            ? std::min(InBlockCopySrcDataPerRead_B, GetReadWriteVectorSize(ctx.kernel_dilation_w))
            : InBlockCopySrcDataPerRead_B;
    InBlockCopySrcDataPerRead_B = ctx.kernel_stride_w > 1 ? 1 : InBlockCopySrcDataPerRead_B;

    // clang-format off
    construction_parameters.comp_options = 
        std::string(" -std=c++14 ") +
        std::string(" -DCK_PARAM_PROBLEM_N=") + std::to_string(n) +
        std::string(" -DCK_PARAM_PROBLEM_K=") + std::to_string(k) +
        std::string(" -DCK_PARAM_PROBLEM_C=") + std::to_string(c) +
        std::string(" -DCK_PARAM_PROBLEM_HI=") + std::to_string(hi) +
        std::string(" -DCK_PARAM_PROBLEM_WI=") + std::to_string(wi) +
        std::string(" -DCK_PARAM_PROBLEM_HO=") + std::to_string(ho) +
        std::string(" -DCK_PARAM_PROBLEM_WO=") + std::to_string(wo) +
        std::string(" -DCK_PARAM_PROBLEM_Y=") + std::to_string(y) +
        std::string(" -DCK_PARAM_PROBLEM_X=") + std::to_string(x) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(conv_stride_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(conv_stride_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_H=") + std::to_string(conv_dilation_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DILATION_W=") + std::to_string(conv_dilation_w) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_H=") + std::to_string(left_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_LEFT_PAD_W=") + std::to_string(left_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_H=") + std::to_string(right_pad_h) +
        std::string(" -DCK_PARAM_PROBLEM_RIGHT_PAD_W=") + std::to_string(right_pad_w) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_FORWARD=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_DATA=") + std::to_string(0) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_DIRECTION_BACKWARD_WEIGHT=") + std::to_string(1) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_B_PER_BLOCK=") + std::to_string(b_per_block) +
        std::string(" -DCK_PARAM_TUNABLE_K_PER_BLOCK=") + std::to_string(k_per_block) +
        std::string(" -DCK_PARAM_TUNABLE_E_PER_BLOCK=") + std::to_string(e_per_block) +
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
        std::string(" -DCK_PARAM_IN_BLOCK_COPY_DST_DATA_PER_WRITE_N2=") + std::to_string(InBlockCopyDstDataPerWrite_N2) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_E=") + std::to_string(config.WeiBlockCopyClusterLengths_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_CLUSTER_LENGTHS_K=") + std::to_string(config.WeiBlockCopyClusterLengths_K) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_SRC_DATA_PER_READ_E=") + std::to_string(WeiBlockCopySrcDataPerRead_E) +
        std::string(" -DCK_PARAM_WEI_BLOCK_COPY_DST_DATA_PER_WRITE_K=") + std::to_string(WeiBlockCopyDstDataPerWrite_K) +
        std::string(" -DCK_PARAM_EPACK_LENGTH=") + std::to_string(GetEPackLength(ctx, false)) + 
        std::string(" -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx)? '1' : '0') +
        std::string(" -D__HIP_PLATFORM_HCC__=1") +
        ctx.general_compile_options;
    // clang-format on

    result.construction_params.push_back(construction_parameters);
    return result;
}

} // namespace solver
} // namespace miopen
