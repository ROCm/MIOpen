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

void PerformanceConvMlirIgemmXdlops::SetMlirHeuristicInitRequest()
{
    // These values are equivalent to when tuning config is heuristically initialized.
    // We leave all config fields to be -2/false and use_spare_set untouched.
    GemmMPerBlock                = -2;
    GemmNPerBlock                = -2;
    GemmKPerBlock                = -2;
    GemmMPerWave                 = -2;
    GemmNPerWave                 = -2;
    GemmKPACKSize                = -2;
    GemmAThreadCopyMoreGemmK     = false;
    GemmBThreadCopyMoreGemmKPack = false;
}

bool ConvMlirIgemmFwdXdlops::IsApplicable(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_MLIR_IGEMM_FWD_XDLOPS{}))
        return false;
    if(miopen::IsEnabled(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC{}))
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
    if(spare)
        SetMlirHeuristicInitRequest();
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
        && GemmBThreadCopyMoreGemmKPack  == other.GemmBThreadCopyMoreGemmKPack;
    // clang-format on
}

bool PerformanceConvMlirIgemmXdlops::IsValid(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    if(*this == MlirHeuristicInitRequest())
        return true;

    int kernel_count = MiirGetKernelCount(mlir::ConstructBuildOptions(ctx, true));
    bool isValid     = false;
    for(int kernel_id = 0; kernel_id < kernel_count; ++kernel_id)
    {
        isValid = MiirIsConfigApplicable(mlir::ConstructBuildOptions(ctx, *this, true, kernel_id));
        if(!isValid)
            return false;
    }
    return isValid;
#else
    std::ignore = ctx;
    return false;
#endif
}

bool PerformanceConvMlirIgemmXdlops::SetNextValue(const ConvolutionContext& config)
{
    if(use_spare_set)
        return false;

    GemmBThreadCopyMoreGemmKPack = true;
    GemmAThreadCopyMoreGemmK     = true;
    do
    {
        if(!NextTwoPower<4, 256>(GemmMPerBlock))
            break;
        if(!NextTwoPower<16, 256>(GemmNPerBlock))
            break;
        if(!NextTwoPower<4, 128>(GemmMPerWave))
            break;
        if(!NextTwoPower<16, 128>(GemmNPerWave))
            break;
        if(!NextTwoPower<4, 8>(GemmKPACKSize))
            break;

        if(config.IsInt8())
        {
            // xdlops instructions supported with in8 determines the minimum valid kPerBlock is 8
            if(!NextTwoPower<8, 32>(GemmKPerBlock))
                break;
        }
        else
        {
            if(!NextTwoPower<1, 8>(GemmKPerBlock))
                break;
        }

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
ConvMlirIgemmFwdXdlops::GetDefaultPerformanceConfig(const ConvolutionContext& ctx) const
{
    std::ignore = ctx;
    return PerformanceConvMlirIgemmXdlops::MlirHeuristicInitRequest();
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
                                                 const PerformanceConvMlirIgemmXdlops& config) const
{
#if MIOPEN_USE_MLIR
    ConvSolution result;
    KernelInfo construction_parameters;

    construction_parameters.kernel_name  = mlir::GetKernelName(ctx, true);
    construction_parameters.kernel_file  = construction_parameters.kernel_name + ".mlir";
    construction_parameters.comp_options = mlir::ConstructBuildOptions(ctx, config, true);

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
