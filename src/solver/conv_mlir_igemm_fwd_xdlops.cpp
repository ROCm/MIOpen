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

#include <miopen/conv/invokers/mlir_impl_gemm.hpp>
#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/mlir_build.hpp>
#include <miopen/solver.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/mlir_common.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_MLIR_IGEMM_FWD_XDLOPS)

namespace miopen {
namespace solver {

bool ConvMlirIgemmFwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_MLIR_IGEMM_FWD_XDLOPS{}))
        return false;
    if(!IsXdlopsSupport(ctx))
        return false;
    if(!ctx.direction.IsForward())
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    return MiirIsConfigApplicable(mlir::ConstructBuildOptions(ctx, true));
#else
    std::ignore = ctx;
    return false;
#endif
}

PerformanceConvMlirIgemmXdlops::PerformanceConvMlirIgemmXdlops(int GemmMPerBlock_,
                                                               int GemmNPerBlock_,
                                                               int GemmKPerBlock_,
                                                               int GemmMPerWave_,
                                                               int GemmNPerWave_,
                                                               int GemmKPACKSize_,
                                                               bool GemmAThreadCopyMoreGemmK_,
                                                               bool GemmBThreadCopyMoreGemmKPack_,
                                                               bool use_spare_set_)
    : GemmMPerBlock(GemmMPerBlock_),
      GemmNPerBlock(GemmNPerBlock_),
      GemmKPerBlock(GemmKPerBlock_),
      GemmMPerWave(GemmMPerWave_),
      GemmNPerWave(GemmNPerWave_),
      GemmKPACKSize(GemmKPACKSize_),
      GemmAThreadCopyMoreGemmK(GemmAThreadCopyMoreGemmK_),
      GemmBThreadCopyMoreGemmKPack(GemmBThreadCopyMoreGemmKPack_),
      use_spare_set(use_spare_set_)
{
}

PerformanceConvMlirIgemmXdlops::PerformanceConvMlirIgemmXdlops(bool spare)
    : PerformanceConvMlirIgemmXdlops::PerformanceConvMlirIgemmXdlops(
          4, 16, 1, 4, 16, 4, false, false, spare)
{
}

PerformanceConvMlirIgemmXdlops::PerformanceConvMlirIgemmXdlops()
    : PerformanceConvMlirIgemmXdlops::PerformanceConvMlirIgemmXdlops(
          -1, -1, -1, -1, -1, -1, false, false)
{
}

bool PerformanceConvMlirIgemmXdlops::operator==(const PerformanceConvMlirIgemmXdlops& other) const
{
    // clang-format off
    return GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmMPerWave == other.GemmMPerWave
        && GemmNPerWave == other.GemmNPerWave
        && GemmKPACKSize == other.GemmKPACKSize
        && GemmAThreadCopyMoreGemmK  == other.GemmAThreadCopyMoreGemmK
        && GemmBThreadCopyMoreGemmKPack  == other.GemmBThreadCopyMoreGemmKPack
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

bool PerformanceConvMlirIgemmXdlops::IsValid(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    bool isValid = MiirIsConfigApplicable(mlir::ConstructBuildOptions(ctx, ToString(), true));
    return isValid;
#else
    std::ignore = ctx;
    return false;
#endif
}

bool PerformanceConvMlirIgemmXdlops::SetNextValue(const ConvolutionContext& /*config*/)
{
    GemmBThreadCopyMoreGemmKPack = true;
    GemmAThreadCopyMoreGemmK     = true;
    do
    {
        if(!NextTwoPower<4, 256>(GemmMPerBlock))
            break;
        if(!NextTwoPower<16, 256>(GemmNPerBlock))
            break;
        if(!NextTwoPower<1, 8>(GemmKPerBlock))
            break;
        if(!NextTwoPower<4, 128>(GemmMPerWave))
            break;
        if(!NextTwoPower<16, 128>(GemmNPerWave))
            break;
        if(!NextTwoPower<4, 8>(GemmKPACKSize))
            break;

        return false;
    } while(false);

    return true;
}

std::string PerformanceConvMlirIgemmXdlops::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConvMlirIgemmXdlops
ConvMlirIgemmFwdXdlops::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    std::ignore = ctx;
    return {};
}

bool ConvMlirIgemmFwdXdlops::IsValidPerformanceConfig(
    const ConvolutionContext& ctx, const PerformanceConvMlirIgemmXdlops& config) const
{
    MIOPEN_LOG_I("");
    return config.IsValid(ctx);
}

PerformanceConvMlirIgemmXdlops
ConvMlirIgemmFwdXdlops::Search(const ConvolutionContext& ctx,
                               const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, invoke_ctx);
}

ConvSolution ConvMlirIgemmFwdXdlops::GetSolution(const ConvolutionContext& ctx,
                                                 const PerformanceConvMlirIgemmXdlops& config,
                                                 bool) const
{
#if MIOPEN_USE_MLIR
    ConvSolution result;
    KernelInfo construction_parameters;

    construction_parameters.kernel_name = mlir::GetKernelName(ctx, true);
    construction_parameters.kernel_file = construction_parameters.kernel_name + ".mlir";

    if(config == PerformanceConvMlirIgemmXdlops())
        // At this case, do not pass in the invalid perf config and instead make Miir library to do
        // heuristic initialization
        construction_parameters.comp_options = mlir::ConstructBuildOptions(ctx, true);
    else
        // At this case, Make Miir library to use the valid perf config
        construction_parameters.comp_options =
            mlir::ConstructBuildOptions(ctx, config.ToString(), true);

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
    std::ignore = config;
    return {};
#endif
}

} // namespace solver
} // namespace miopen
