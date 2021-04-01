/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#include <miopen/mlir_build.hpp>
#include <miopen/conv/invokers/mlir_impl_gemm.hpp>
#include <miopen/solver.hpp>
#include <miopen/handle.hpp>

#include "implicitgemm_util.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_HIP_IMPLICIT_GEMM_MLIR_BIN_FWD)

namespace miopen {
namespace solver {

std::tuple<int, int, int>
ConvHipImplicitGemmMlirBinFwd::CalculateGemmSize(const ConvolutionContext& ctx)
{
    const auto n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const auto k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const auto c  = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const auto ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const auto wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const auto y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const auto x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

    const auto gemm_m = k;
    const auto gemm_n = ctx.Is3d()
                            ? n * ho * wo * ConvolutionContextInterpreter::GetOutputDepthDo(ctx)
                            : n * ho * wo;
    const auto gemm_k =
        ctx.Is3d() ? c * y * x * ConvolutionContextInterpreter::GetFilterDepthZ(ctx) : c * y * x;

    return std::make_tuple(gemm_m, gemm_n, gemm_k);
}

bool ConvHipImplicitGemmMlirBinFwd::IsApplicable(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_HIP_IMPLICIT_GEMM_MLIR_BIN_FWD{}))
        return false;
    // Future: MLIR-binary solutions do not take long time to build
    if(ctx.skip_solutions_that_take_long_time_to_build_and_have_narrow_coverage)
        return false;
    // Future: MLIR will support non-default layouts.
    if(!ctx.IsLayoutDefault())
        return false;
    // Future: MLIR will support 3d convolution
    if(!ctx.Is2d())
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    if(!ctx.direction.IsForward())
        return false;
    if(!ctx.IsFp32())
        return false;
    if(ctx.group_counts != 1)
        return false;

    int gemm_m = 0;
    int gemm_n = 0;
    int gemm_k = 0;

    std::tie(gemm_m, gemm_n, gemm_k) = CalculateGemmSize(ctx);
    return gemm_m % 32 == 0 && gemm_n % 32 == 0 && gemm_k % 4 == 0;
#else
    std::ignore = ctx;
    return false;
#endif
}

ConvSolution ConvHipImplicitGemmMlirBinFwd::GetSolution(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    ConvSolution result;
    KernelInfo construction_parameters;

    std::string version   = "_v4r4";
    std::string direction = "_fwd";
    std::string operation = "conv2d";

    construction_parameters.kernel_file = "mlir_gen_igemm_conv2d" + version + direction + ".mlir";

    construction_parameters.kernel_name = "mlir_gen_igemm_conv2d" + version + direction;

    // Arguments for mlir-miopen-driver.
    // clang-format off
    using CI = ConvolutionContextInterpreter;
    construction_parameters.comp_options =
        std::string(" --operation ") + operation +
        std::string(" --num_cu ") + std::to_string(ctx.GetStream().GetMaxComputeUnits()) +
        std::string(" --arch ") + ctx.GetStream().GetDeviceName() +
        std::string(" --fil_layout ") + CI::GetFilterLayout(ctx) +
        std::string(" --fil_type ") + "fp32" +
        std::string(" --in_layout ") + CI::GetInputLayout(ctx) +
        std::string(" --in_type ") + "fp32" +
        std::string(" --out_layout ") + CI::GetOutputLayout(ctx) +
        std::string(" --out_type ") + "fp32" +
        std::string(" --batchsize ") + std::to_string(CI::GetBatchN(ctx)) +
        std::string(" --in_channels ") + std::to_string(CI::GetInputChannelC(ctx)) +
        std::string(" --out_channels ") + std::to_string(CI::GetOutputChannelK(ctx)) +
        std::string(" --in_h ") + std::to_string(CI::GetInputHeightHi(ctx)) +
        std::string(" --in_w ") + std::to_string(CI::GetInputWidthWi(ctx)) +
        std::string(" --out_h ") + std::to_string(CI::GetOutputHeightHo(ctx)) +
        std::string(" --out_w ") + std::to_string(CI::GetOutputWidthWo(ctx)) +
        std::string(" --fil_h ") + std::to_string(CI::GetFilterHeightY(ctx)) +
        std::string(" --fil_w ") + std::to_string(CI::GetFilterWidthX(ctx)) +
        std::string(" --dilation_h ") + std::to_string(CI::GetAdjustedConvolutionDilationH(ctx)) +
        std::string(" --dilation_w ") + std::to_string(CI::GetAdjustedConvolutionDilationW(ctx)) +
        std::string(" --conv_stride_h ") + std::to_string(CI::GetAdjustedConvolutionStrideH(ctx)) +
        std::string(" --conv_stride_w ") + std::to_string(CI::GetAdjustedConvolutionStrideW(ctx)) +
        std::string(" --padding_h ") + std::to_string(CI::GetInputLeftPadH(ctx)) +
        std::string(" --padding_w ") + std::to_string(CI::GetInputLeftPadW(ctx)) +
        std::string(" --kernel_name ") + construction_parameters.kernel_name;
    // clang-format on

    size_t local_size  = 0;
    size_t global_size = 0;
    MiirGenLaunchParams(construction_parameters.comp_options, local_size, global_size);

    construction_parameters.l_wk.push_back(local_size);
    construction_parameters.l_wk.push_back(1);
    construction_parameters.l_wk.push_back(1);

    construction_parameters.g_wk.push_back(global_size);
    construction_parameters.g_wk.push_back(1);
    construction_parameters.g_wk.push_back(1);

    result.invoker_factory = conv::MakeMlirFwdInvokerFactory(ctx);
    result.construction_params.push_back(construction_parameters);
    return result;
#else
    std::ignore = ctx;
    return {};
#endif
}

} // namespace solver
} // namespace miopen
