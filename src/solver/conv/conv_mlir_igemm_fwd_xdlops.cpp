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
#include <miopen/conv/solvers.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/mlir_common.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_MLIR_IGEMM_FWD_XDLOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

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

bool ConvMlirIgemmFwdXdlops::IsApplicable(const ExecutionContext& ctx,
                                          const ProblemDescription& problem) const
{
#if MIOPEN_USE_MLIR
    if(env::disabled(MIOPEN_DEBUG_CONV_MLIR_IGEMM_FWD_XDLOPS))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(!IsXdlopsSupport(ctx))
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    if(problem.IsTensorsCasted() || problem.IsFp8() || problem.IsBfp8())
        return false;
    return MiirIsConfigApplicable(mlir::ConstructBuildOptions(ctx, problem, true));
#else
    std::ignore = ctx;
    std::ignore = problem;
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

bool PerformanceConvMlirIgemmXdlops::IsValid(const ExecutionContext& ctx,
                                             const ProblemDescription& problem) const
{
#if MIOPEN_USE_MLIR
    if(*this == MlirHeuristicInitRequest())
        return true;

    int kernel_count = MiirGetKernelCount(mlir::ConstructBuildOptions(ctx, problem, *this, true));
    bool isValid     = false;
    for(int kernel_id = 0; kernel_id < kernel_count; ++kernel_id)
    {
        isValid = MiirIsConfigApplicable(
            mlir::ConstructBuildOptions(ctx, problem, *this, true, kernel_id));
        if(!isValid)
            return false;
    }
    return isValid;
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

bool PerformanceConvMlirIgemmXdlops::SetNextValue(const ProblemDescription& problem)
{
    if(use_spare_set)
        return false;

    GemmBThreadCopyMoreGemmKPack = true;
    do
    {
        if(!NextFlag<false, true>(GemmAThreadCopyMoreGemmK))
            break;
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

        if(problem.IsInt8())
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

PerformanceConvMlirIgemmXdlops
ConvMlirIgemmFwdXdlops::GetDefaultPerformanceConfig(const ExecutionContext&,
                                                    const ProblemDescription&) const
{
    return PerformanceConvMlirIgemmXdlops::MlirHeuristicInitRequest();
}

bool ConvMlirIgemmFwdXdlops::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConvMlirIgemmXdlops& config) const
{
    MIOPEN_LOG_I("");
    return config.IsValid(ctx, problem);
}

PerformanceConvMlirIgemmXdlops
ConvMlirIgemmFwdXdlops::Search(const ExecutionContext& ctx,
                               const ProblemDescription& problem,
                               const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution ConvMlirIgemmFwdXdlops::GetSolution(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem,
                                                 const PerformanceConvMlirIgemmXdlops& config) const
{
#if MIOPEN_USE_MLIR
    ConvSolution result;
    KernelInfo construction_parameters;

    construction_parameters.kernel_name  = mlir::GetKernelName(problem, true);
    construction_parameters.kernel_file  = construction_parameters.kernel_name + ".mlir";
    construction_parameters.comp_options = mlir::ConstructBuildOptions(ctx, problem, config, true);

    size_t local_size  = 0;
    size_t global_size = 0;
    MiirGenLaunchParams(construction_parameters.comp_options, local_size, global_size);

    construction_parameters.l_wk.push_back(local_size);
    construction_parameters.l_wk.push_back(1);
    construction_parameters.l_wk.push_back(1);

    construction_parameters.g_wk.push_back(global_size);
    construction_parameters.g_wk.push_back(1);
    construction_parameters.g_wk.push_back(1);

    result.invoker_factory = miopen::conv::MakeMlirFwdInvokerFactory(problem);
    result.construction_params.push_back(construction_parameters);
    return result;
#else
    std::ignore = ctx;
    std::ignore = problem;
    std::ignore = config;
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
