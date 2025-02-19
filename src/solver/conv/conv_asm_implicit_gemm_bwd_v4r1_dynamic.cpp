/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include "miopen/conv/solvers.hpp"
#include "miopen/handle.hpp"
#include <miopen/conv/invokers/impl_gemm_dynamic.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <algorithm>
#include <miopen/solver/implicitgemm_util.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_V4R1)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

static inline bool FindImplicitGemmDynamicKernelBwd(const ProblemDescription& problem,
                                                    std::string& kernel_name,
                                                    int& block_size,
                                                    int& grid_size)
{
    // TODO: add more dynamic kernel to expand support range, and update this function
    // clang-format off
    // refer to ProblemInterpreter, in bwd most dimension is reversed
    int hi          = problem.GetOutHeight();
    int wi          = problem.GetOutWidth();
    int n           = problem.GetBatchSize();
    int k           = problem.GetInChannels();
    int c           = problem.GetOutChannels();
    int ho          = problem.GetInHeight();
    int wo          = problem.GetInWidth();
    int stride_h    = problem.GetInHeight() > 1 ? problem.GetKernelStrideH() : 1;
    int stride_w    = problem.GetInWidth() > 1 ? problem.GetKernelStrideW() : 1;
    int dilation_h  = problem.GetWeightsHeight() > 1? problem.GetDilationH() : 1;
    int dilation_w  = problem.GetWeightsWidth() > 1? problem.GetDilationW() : 1;
    int pad_h       = problem.GetPadH();
    int pad_w       = problem.GetPadW();
    int y           = problem.GetWeightsHeight();
    int x           = problem.GetWeightsWidth();

    int gcd_stride_dilation_h = gcd(stride_h, dilation_h);
    int gcd_stride_dilation_w = gcd(stride_w, dilation_w);
    int y_tilda     = stride_h / gcd_stride_dilation_h;
    int x_tilda     = stride_w / gcd_stride_dilation_w;

    int h_tilda     = ho + (dilation_h * (y - 1) + stride_h - 1) / stride_h;
    int w_tilda     = wo + (dilation_w * (x - 1) + stride_w - 1) / stride_w;

    int h_tilda_left = std::max(0, pad_h - dilation_h * (y_tilda - 1)) / stride_h;
    int w_tilda_left = std::max(0, pad_w - dilation_w * (x_tilda - 1)) / stride_w;

    int h_tilda_right = std::min(h_tilda, (pad_h + hi - 1 + stride_h - 1) / stride_h + 1);
    int w_tilda_right = std::min(w_tilda, (pad_w + wi - 1 + stride_w - 1) / stride_w + 1);

    int h_tilda_slice = h_tilda_right - h_tilda_left;
    int w_tilda_slice = w_tilda_right - w_tilda_left;
    // clang-format on
    int gemm_m = c;
    int gemm_n = n * h_tilda_slice * w_tilda_slice;
    // int gemm_k; since k dimension is merged, we only check k

    // TODO: this is too simple, need more kernels and more optimal logic to select kernel
    if((gemm_m % 128 == 0) && (gemm_n % 128 == 0) && (k % 16 == 0))
    {
        if((y == 1) && (x == 1) && (stride_h == 1) && (stride_w == 1) && (dilation_h == 1) &&
           (dilation_w == 1) && (pad_h == 0) && (pad_w == 0) && (n % 128 == 0))
        {
            grid_size   = (gemm_m >> 7) * (gemm_n >> 7);
            block_size  = 256;
            kernel_name = "igemm_bwd_gtc_bt128x128x16_tt8x8_gm2x4x4_gn2x4x4_ta1x1x1x2x4_"
                          "16x1x1x16x1_tb1x1x1x2x4x1x1_16x1x1x16x1x1x1";
            return true;
        }
        else
        {
            grid_size   = (gemm_m >> 7) * (gemm_n >> 7);
            block_size  = 256;
            kernel_name = "igemm_bwd_gtc";
            return true;
        }
    }
    else
    {
        if((y == 1) && (x == 1) && (stride_h == 1) && (stride_w == 1) && (dilation_h == 1) &&
           (dilation_w == 1) && (pad_h == 0) && (pad_w == 0))
        {
            if((gemm_m % 128 == 0) && (gemm_n % 128 == 0) && (k % 8 == 0) && ((ho * wo) % 16 == 0))
            {
                grid_size   = (gemm_m >> 7) * (gemm_n >> 7);
                block_size  = 256;
                kernel_name = "igemm_bwd_gtc_bt128x128x8_tt8x8_gm2x4x4_gn2x4x4_ta1x1x1x1x4_"
                              "8x1x1x32x1_tb1x1x1x1x4x1x1_8x1x1x2x1x1x16";
                return true;
            }
            else if((gemm_m % 64 == 0) && (gemm_n % 64 == 0) && (k % 8 == 0) && (n % 64 == 0))
            {
                grid_size   = (gemm_m >> 6) * (gemm_n >> 6);
                block_size  = 64;
                kernel_name = "igemm_bwd_gtc_bt64x64x8_tt8x8_gm2x4x2_gn2x4x2_ta1x2x1x1x4_"
                              "4x1x1x16x1_tb1x2x1x1x4x1x1_4x1x1x16x1x1x1";
                return true;
            }
        }
    }
    return false;
}

bool ConvAsmImplicitGemmV4R1DynamicBwd::IsApplicable(const ExecutionContext& ctx,
                                                     const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_BWD_V4R1))
        return false;

    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx900") || StartsWith(device_name, "gfx906")))
        return false;

    if(!ctx.use_asm_kernels)
        return false;

    if(!problem.IsDirectionBackwardData())
        return false;

    if(problem.HasNonPackedTensors())
        return false;

    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    if(!problem.Is2d())
        return false;

    if(!problem.IsFp32())
        return false;

    if(problem.IsTensorsCasted())
        return false;

    if(!ctx.rmv.IsV3())
        return false;

    if(problem.GetGroupCount() != 1)
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    const auto target = ctx.GetStream().GetTargetProperties();
    if(target.Xnack() && *target.Xnack())
        return false;

    std::string kernel_name;
    int block_size;
    int grid_size;
    return FindImplicitGemmDynamicKernelBwd(problem, kernel_name, block_size, grid_size);
}

ConvSolution ConvAsmImplicitGemmV4R1DynamicBwd::GetSolution(const ExecutionContext& ctx,
                                                            const ProblemDescription& problem) const
{
    ConvSolution result;

    std::string kernel_name;
    int block_size;
    int grid_size;
    bool ret = FindImplicitGemmDynamicKernelBwd(problem, kernel_name, block_size, grid_size);
    if(!ret)
        MIOPEN_THROW("should not happen!");

    KernelInfo kernel;
    std::ostringstream options;

    kernel.kernel_file = "igemm_bwd_gtc_dynamic.s";
    kernel.kernel_name = kernel_name;
    kernel.g_wk.clear();
    /* Note here, for API like hipHccModuleLaunchKernel(), hipExtModuleLaunchKernel()
     * grid dims is in unit of work item.
     * But for api like hipModuleLaunchKernel(), grid dim is in unit of block.
     */
    kernel.g_wk.push_back(static_cast<std::size_t>(grid_size) * block_size);
    kernel.g_wk.push_back(1);
    kernel.g_wk.push_back(1);
    kernel.l_wk.clear();
    kernel.l_wk.push_back(block_size);
    kernel.l_wk.push_back(1);
    kernel.l_wk.push_back(1);

    GenerateClangDefsym(options, "ROCM_METADATA_VERSION", ctx.rmv.UseV3() ? 5 : 4);

    kernel.comp_options = options.str();

    result.invoker_factory =
        miopen::conv::MakeImplGemmDynamicBackwardDataInvokerFactory(problem, int(0));
    result.construction_params.push_back(kernel);
    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
