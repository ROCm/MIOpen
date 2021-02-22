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
#include <miopen/conv/invokers/impl_gemm.hpp>
#include <miopen/solver.hpp>
#include <miopen/handle.hpp>

#include "implicitgemm_util.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_HIP_IMPLICIT_GEMM_MLIR_CPP_FWD)

namespace miopen {
namespace solver {

bool ConvHipImplicitGemmMlirCppFwd::IsApplicable(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_HIP_IMPLICIT_GEMM_MLIR_CPP_FWD{}))
        return false;
    // Future: MLIR-binary solutions do not take long time to build
    if(ctx.skip_solutions_that_take_long_time_to_build_and_have_narrow_coverage)
        return false;
    // Future: MLIR-binary solutions do not use HIP kernels.
    if(!ctx.use_hip_kernels)
        return false;
    // Future: MLIR will support non-default layouts.
    if(!ctx.IsLayoutDefault())
        return false;
    if(ctx.Is3d())
        return false;
    return IsApplicableMlirCommon(ctx);
#else
    std::ignore = ctx;
    return false;
#endif
}

ConvSolution ConvHipImplicitGemmMlirCppFwd::GetSolution(const ConvolutionContext& ctx) const
{
    ConvSolution result;
    KernelInfo construction_parameters;

    int grid_size = 0;

    const auto config = GetPerformanceConfig(ctx);
    std::tie(grid_size, std::ignore) = config.CalculateGridSize(ctx);

    construction_parameters.l_wk.push_back(config.BlockSize);
    construction_parameters.l_wk.push_back(1);
    construction_parameters.l_wk.push_back(1);

    construction_parameters.g_wk.push_back(config.BlockSize * grid_size);
    construction_parameters.g_wk.push_back(1);
    construction_parameters.g_wk.push_back(1);

    std::string version   = "_v4r4";
    std::string direction = "_fwd";
    std::string operation = "conv2d";

    construction_parameters.kernel_file =
        "mlir_gen_igemm_conv2d_cpp" + version + direction + ".mlir-cpp";

    construction_parameters.kernel_name = "mlir_gen_igemm_conv2d_cpp" + version + direction;

    // Arguments for mlir-miopen-driver.
    // clang-format off
    using CI = ConvolutionContextInterpreter;
    construction_parameters.comp_options =
        std::string(" --operation ") + operation +
        std::string(" --fil_layout ") + CI::GetFilterLayout(ctx) +
        std::string(" --in_layout ") + CI::GetInputLayout(ctx) +
        std::string(" --out_layout ") + CI::GetOutputLayout(ctx) +
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

    result.invoker_factory = conv::MakeImplGemmDataInvokerFactory(ctx);
    result.construction_params.push_back(construction_parameters);
    return result;
}

} // namespace solver
} // namespace miopen
