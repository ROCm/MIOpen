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
#include <numeric>
#include "miopen/solver.hpp"
#include "miopen/handle.hpp"
#include <miopen/generic_search.hpp>
#include "implicitgemm_util.hpp"

namespace miopen {
namespace solver {

// greatest common divisor, aka highest common factor
template <typename T>
constexpr T gcd(T x, T y)
{
    if(x == 0)
    {
        return y;
    }

    if(y == 0)
    {
        return x;
    }

    if(x == y)
    {
        return x;
    }

    if(x > y)
    {
        return gcd(x - y, y);
    }

    return gcd(x, y - x);
}

template <typename X, typename... Ys>
constexpr auto gcd(X x, Ys... ys)
{
    return gcd(x, ys...);
}

template <typename T>
static inline T integer_divide_ceil(T x, T y)
{
    return (x + y - 1) / y;
}

static inline std::size_t get_number_of_gemm(const ConvolutionContext& ctx)
{
    std::size_t conv_stride_h = KernelFilterStrideH(ctx);
    std::size_t conv_stride_w = KernelFilterStrideW(ctx);

    std::size_t conv_dilation_h = KernelFilterDilationH(ctx);
    std::size_t conv_dilation_w = KernelFilterDilationW(ctx);

    std::size_t gcd_stride_dilation_h = gcd(conv_stride_h, conv_dilation_h);
    std::size_t gcd_stride_dilation_w = gcd(conv_stride_w, conv_dilation_w);

    std::size_t ytilda = conv_stride_h / gcd_stride_dilation_h;
    std::size_t xtilda = conv_stride_w / gcd_stride_dilation_w;

    return ytilda * xtilda;
}

static inline auto get_gemm_size(const ConvolutionContext& ctx, int gemm_id)
{
    int n               = ConvolutionContextInterpreter::GetBatchN(ctx);
    int k               = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    int c               = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    int hi              = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
    int wi              = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
    int ho              = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    int wo              = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    int y               = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    int x               = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
    int conv_stride_h   = ConvolutionContextInterpreter::GetConvolutionStrideH(ctx);
    int conv_stride_w   = ConvolutionContextInterpreter::GetConvolutionStrideW(ctx);
    int conv_dilation_h = ConvolutionContextInterpreter::GetConvolutionDilationH(ctx);
    int conv_dilation_w = ConvolutionContextInterpreter::GetConvolutionDilationW(ctx);
    int in_left_pad_h   = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
    int in_left_pad_w   = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);

    int gcd_stride_dilation_h = gcd(conv_stride_h, conv_dilation_h);
    int gcd_stride_dilation_w = gcd(conv_stride_w, conv_dilation_w);

    int ytilda = conv_stride_h / gcd_stride_dilation_h;
    int xtilda = conv_stride_w / gcd_stride_dilation_w;

    int ydot = integer_divide_ceil(y, ytilda);
    int xdot = integer_divide_ceil(x, xtilda);

    int htilda = ho + integer_divide_ceil(conv_dilation_h * (y - 1), conv_stride_h);
    int wtilda = wo + integer_divide_ceil(conv_dilation_w * (x - 1), conv_stride_w);

    // intermediate result could be negative, use int instead of size_t
    int htilda_left = std::max(0, in_left_pad_h - conv_dilation_h * (ytilda - 1)) / conv_stride_h;
    int wtilda_left = std::max(0, in_left_pad_w - conv_dilation_w * (xtilda - 1)) / conv_stride_w;

    int htilda_right =
        std::min(htilda, integer_divide_ceil(in_left_pad_h + hi - 1, conv_stride_h) + 1);
    int wtilda_right =
        std::min(wtilda, integer_divide_ceil(in_left_pad_w + wi - 1, conv_stride_w) + 1);

    int htilda_slice = htilda_right - htilda_left;
    int wtilda_slice = wtilda_right - wtilda_left;

    // gemm_k size is different for each GEMM
    int i_ytilda = gemm_id / xtilda;
    int i_xtilda = gemm_id % xtilda;

    int ydot_slice = (i_ytilda + 1) * ydot <= y ? ydot : y % ydot;
    int xdot_slice = (i_xtilda + 1) * xdot <= x ? xdot : x % xdot;

    std::size_t gemm_m = static_cast<std::size_t>(c);
    std::size_t gemm_n = static_cast<std::size_t>(n) * htilda_slice * wtilda_slice;
    std::size_t gemm_k = static_cast<std::size_t>(k) * ydot_slice * xdot_slice;

    return std::make_tuple(gemm_m, gemm_n, gemm_k);
}

bool ConvHipImplicitGemmBwdDataV4R1::IsApplicable(const ConvolutionContext& ctx) const
{
    bool is_applicable = true;

    if(!ctx.direction.IsBackwardData())
        return false;

    if(!ctx.Is2d())
        return false;

    if(!ctx.IsFp32())
        return false;

    if(ctx.group_counts != 1)
        return false;

    std::size_t gemm_m = 0;
    std::size_t gemm_n = 0;

    std::tie(gemm_m, gemm_n, std::ignore) = get_gemm_size(ctx, 0);

    is_applicable = is_applicable && gemm_m % 128 == 0 && gemm_n % 128 == 0;

    for(int i = 0; i < get_number_of_gemm(ctx); ++i)
    {
        std::size_t gemm_k = 0;

        std::tie(std::ignore, std::ignore, gemm_k) = get_gemm_size(ctx, i);

        is_applicable = is_applicable && gemm_k % 8 == 0;
    }

    return is_applicable;
}

ConvSolution ConvHipImplicitGemmBwdDataV4R1::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;

    // a series of kernels
    for(std::size_t gemm_id = 0; gemm_id < get_number_of_gemm(ctx); ++gemm_id)
    {
        KernelInfo construction_parameters;

        std::size_t n               = ConvolutionContextInterpreter::GetBatchN(ctx);
        std::size_t k               = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
        std::size_t c               = ConvolutionContextInterpreter::GetInputChannelC(ctx);
        std::size_t hi              = ConvolutionContextInterpreter::GetInputHeightHi(ctx);
        std::size_t wi              = ConvolutionContextInterpreter::GetInputWidthWi(ctx);
        std::size_t ho              = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
        std::size_t wo              = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
        std::size_t y               = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
        std::size_t x               = ConvolutionContextInterpreter::GetFilterWidthX(ctx);
        std::size_t conv_stride_h   = ConvolutionContextInterpreter::GetConvolutionStrideH(ctx);
        std::size_t conv_stride_w   = ConvolutionContextInterpreter::GetConvolutionStrideW(ctx);
        std::size_t conv_dilation_h = ConvolutionContextInterpreter::GetConvolutionDilationH(ctx);
        std::size_t conv_dilation_w = ConvolutionContextInterpreter::GetConvolutionDilationW(ctx);
        std::size_t in_left_pad_h   = ConvolutionContextInterpreter::GetInputLeftPadH(ctx);
        std::size_t in_left_pad_w   = ConvolutionContextInterpreter::GetInputLeftPadW(ctx);
        std::size_t in_right_pad_h  = ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx);
        std::size_t in_right_pad_w  = ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx);

        std::size_t gemm_m = 0;
        std::size_t gemm_n = 0;
        std::size_t gemm_k = 0;

        std::tie(gemm_m, gemm_n, gemm_k) = get_gemm_size(ctx, gemm_id);

        // don't compile or launch an empty gridwise GEMM
        if(gemm_k > 0)
        {
            std::size_t gemm_m_per_block = 128;
            std::size_t gemm_n_per_block = 128;

            std::size_t block_size = 256;

            std::size_t grid_size = (gemm_m / gemm_m_per_block) * (gemm_n / gemm_n_per_block);

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
                "gridwise_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw.cpp";

            construction_parameters.kernel_name =
                "gridwise_convolution_backward_data_implicit_gemm_v4r1_nchw_kcyx_nkhw";

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
                std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_H=") + std::to_string(in_left_pad_h) +
                std::string(" -DCK_PARAM_PROBLEM_IN_LEFT_PAD_W=") + std::to_string(in_left_pad_w) +
                std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_H=") + std::to_string(in_right_pad_h) +
                std::string(" -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_W=") + std::to_string(in_right_pad_w) +
                std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=") + std::to_string(gemm_m_per_block) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=") + std::to_string(gemm_n_per_block) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=") + std::to_string(8) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_M_PER_THREAD_SUB_C=") + std::to_string(4) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_N_PER_THREAD_SUB_C=") + std::to_string(4) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_M_LEVEL0_CLUSTER=") + std::to_string(4) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_N_LEVEL0_CLUSTER=") + std::to_string(4) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_M_LEVEL1_CLUSTER=") + std::to_string(4) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_N_LEVEL1_CLUSTER=") + std::to_string(4) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(2) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=") + std::to_string(128) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_M=") + std::to_string(1) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M=") + std::to_string(1) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=") + std::to_string(2) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=") + std::to_string(128) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_N=") + std::to_string(1) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N=") + std::to_string(1) +
                std::string(" -DCK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1=") + std::to_string(1) +
                std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
                std::string(" -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=") + (use_amd_inline_asm(ctx) ? '1' : '0') +
                std::string(" -DCK_PARAM_GEMM_ID=") + std::to_string(gemm_id) +
                std::string(" -D__HIP_PLATFORM_HCC__=1") +
                ctx.general_compile_options;
            // clang-format on

            result.construction_params.push_back(construction_parameters);
        }
    }

    return result;
}

} // namespace solver
} // namespace miopen
