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
#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/solver.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/mlir_common.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_MLIR_IGEMM_BWD)

namespace miopen {
namespace solver {

namespace {
#if MIOPEN_USE_MLIR
std::tuple<int, int, int> CalculateGemmSize(const ConvolutionContext& ctx)
{
    const size_t g  = ConvolutionContextInterpreter::GetGroupCountG(ctx);
    const size_t n  = ConvolutionContextInterpreter::GetBatchN(ctx);
    const size_t k  = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    const size_t c  = ConvolutionContextInterpreter::GetInputChannelC(ctx);
    const size_t ho = ConvolutionContextInterpreter::GetOutputHeightHo(ctx);
    const size_t wo = ConvolutionContextInterpreter::GetOutputWidthWo(ctx);
    const size_t y  = ConvolutionContextInterpreter::GetFilterHeightY(ctx);
    const size_t x  = ConvolutionContextInterpreter::GetFilterWidthX(ctx);

    const auto k_per_group = k / g;
    const auto c_per_group = c / g;

    const auto gemm_m       = c_per_group * y * x;
    const auto gemm_n       = n * ho * wo;
    const auto gemm_k_total = k_per_group;

    return std::make_tuple(gemm_m, gemm_n, gemm_k_total);
}

std::string GetKernelName()
{
    std::string version   = "_v4r1";
    std::string direction = "_bwd";
    return "mlir_gen_igemm_conv2d" + version + direction;
}

std::string GetOperation() { return "conv2d_bwd_data"; }
#endif
} // Anonymous namespace

bool ConvMlirIgemmBwd::IsApplicable(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_MLIR_IGEMM_BWD{}))
        return false;
    if(!ctx.IsLayoutDefault() && !ctx.IsLayoutNHWC())
        return false;
    // Future: MLIR will support 3d convolution
    if(!ctx.Is2d())
        return false;
    if(!ctx.IsFp32() && !ctx.IsFp16())
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    if(!ctx.direction.IsBackwardData())
        return false;

    const auto k = ConvolutionContextInterpreter::GetOutputChannelK(ctx);
    if(k % GetEPackLength(ctx, false) != 0)
        return false;

    int gemm_m = 0;
    int gemm_n = 0;
    int gemm_k = 0;

    std::tie(gemm_m, gemm_n, gemm_k) = CalculateGemmSize(ctx);

    if(!(gemm_m % 32 == 0 && gemm_n % 32 == 0 && gemm_k % 4 == 0))
        return false;

    return MiirIsConfigApplicable(
        mlir::ConstructBuildOptions(ctx, GetOperation(), GetKernelName(), false));
#else
    std::ignore = ctx;
    return false;
#endif
}

ConvSolution ConvMlirIgemmBwd::GetSolution(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    ConvSolution result;
    int kernel_count = MiirGetKernelCount(
        mlir::ConstructBuildOptions(ctx, GetOperation(), GetKernelName(), false));

    for(int kernel_id = 0; kernel_id < kernel_count; ++kernel_id)
    {
        KernelInfo construction_parameters;

        construction_parameters.kernel_name  = GetKernelName() + std::to_string(kernel_id);
        construction_parameters.kernel_file  = construction_parameters.kernel_name + ".mlir";
        construction_parameters.comp_options = mlir::ConstructBuildOptions(
            ctx, GetOperation(), construction_parameters.kernel_name, false, kernel_id);

        size_t local_size  = 0;
        size_t global_size = 0;
        MiirGenLaunchParams(construction_parameters.comp_options, local_size, global_size);
        construction_parameters.l_wk.push_back(local_size);
        construction_parameters.l_wk.push_back(1);
        construction_parameters.l_wk.push_back(1);
        construction_parameters.g_wk.push_back(global_size);
        construction_parameters.g_wk.push_back(1);
        construction_parameters.g_wk.push_back(1);

        result.construction_params.push_back(construction_parameters);
    }

    result.invoker_factory = conv::MakeMlirBwdInvokerFactory(ctx);
    return result;
#else
    std::ignore = ctx;
    return {};
#endif
}

} // namespace solver
} // namespace miopen
