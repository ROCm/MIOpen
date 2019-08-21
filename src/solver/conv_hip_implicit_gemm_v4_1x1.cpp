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
#include "miopen/stringutils.hpp"

namespace miopen {
namespace solver {

static inline int ImgHeight(const ConvolutionContext& c)
{
    return c.direction.IsForward() ? c.out_height : c.in_height;
}

static inline int ImgWidth(const ConvolutionContext& c)
{
    return c.direction.IsForward() ? c.out_width : c.in_width;
}

bool ConvHipImplicitGemmV4_1x1::IsApplicable(const ConvolutionContext& ctx) const
{
    return ctx.IsFp32() && ctx.pad_h == 0 && ctx.pad_w == 0 && ctx.group_counts == 1 &&
           ctx.batch_sz % 8 == 0 && (ctx.batch_sz * ImgHeight(ctx) * ImgWidth(ctx)) % 128 == 0 &&
           ctx.n_outputs % 128 == 0 && ctx.kernel_size_h == 1 && ctx.kernel_size_w == 1 &&
           ctx.n_inputs % 16 == 0 && ctx.kernel_dilation_h == 1 && ctx.kernel_dilation_w == 1;
}

ConvSolution ConvHipImplicitGemmV4_1x1::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    assert(ctx.kernel_size_h == 1 && ctx.kernel_size_w == 1);

    std::size_t n  = ctx.batch_sz;
    std::size_t k  = ctx.n_outputs;
    std::size_t c  = ctx.n_inputs;
    std::size_t ho = ctx.out_height;
    std::size_t wo = ctx.out_width;
    std::size_t hi = ctx.in_height;
    std::size_t wi = ctx.in_width;

    std::size_t b_forw = (n * ho * wo) / 8;
    std::size_t b_back = (n * hi * wi) / 8;

    std::size_t b = ctx.direction.IsForward() ? b_forw : b_back;

    std::size_t b_per_block = 16;
    std::size_t k_per_block = 128;
    std::size_t c_per_block = 8;

    std::size_t block_size = 256;

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
        "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer.cpp";

    construction_parameters.kernel_name =
        "gridwise_convolution_implicit_gemm_v4_nchw_kc1x1_nkhw_lds_double_buffer";

    bool use_amd_inline_asm = true;
    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx8"))
        use_amd_inline_asm = false;

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
        std::string(" -DCK_PARAM_PROBLEM_DIRECTION=") + std::to_string(static_cast<int>(ctx.direction.IsForward())) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_H=") + std::to_string(ctx.kernel_stride_h) +
        std::string(" -DCK_PARAM_PROBLEM_CONV_STRIDE_W=") + std::to_string(ctx.kernel_stride_w) +
        std::string(" -DCK_PARAM_TUNABLE_BLOCK_SIZE=") + std::to_string(block_size) +
        std::string(" -DCK_PARAM_TUNABLE_B_PER_BLOCK=") + std::to_string(b_per_block) +
        std::string(" -DCK_PARAM_TUNABLE_K_PER_BLOCK=") + std::to_string(k_per_block) +
        std::string(" -DCK_PARAM_TUNABLE_C_PER_BLOCK=") + std::to_string(c_per_block) +
        std::string(" -DCK_PARAM_DEPENDENT_GRID_SIZE=") + std::to_string(grid_size) +
        std::string(" -DCK_BLOCKWISE_GEMM_USE_AMD_INLINE_ASM=") + std::to_string(use_amd_inline_asm ? 1 : 0) +
        std::string(" -D__HIP_PLATFORM_HCC__=1") +
        ctx.general_compile_options;
    // clang-format on

    result.construction_params.push_back(construction_parameters);
    return result;
}
} // namespace solver
} // namespace miopen
