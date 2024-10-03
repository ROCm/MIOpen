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
#include <miopen/generic_search.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/mlir_common.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_MLIR_IGEMM_FWD)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

void PerformanceConvMlirIgemm::SetMlirHeuristicInitRequest()
{
    // These values are equivalent to when tuning config is heuristically initialized.
    // We leave all config fields to be -2/false and use_spare_set untouched.
    BlockSize      = -2;
    GemmMPerBlock  = -2;
    GemmNPerBlock  = -2;
    GemmKPerBlock  = -2;
    GemmMPerThread = -2;
    GemmNPerThread = -2;
}

PerformanceConvMlirIgemm::PerformanceConvMlirIgemm(int BlockSize_,
                                                   int GemmMPerBlock_,
                                                   int GemmNPerBlock_,
                                                   int GemmKPerBlock_,
                                                   int GemmMPerThread_,
                                                   int GemmNPerThread_,
                                                   bool use_spare_set_)
    : BlockSize(BlockSize_),
      GemmMPerBlock(GemmMPerBlock_),
      GemmNPerBlock(GemmNPerBlock_),
      GemmKPerBlock(GemmKPerBlock_),
      GemmMPerThread(GemmMPerThread_),
      GemmNPerThread(GemmNPerThread_),
      use_spare_set(use_spare_set_)
{
}

PerformanceConvMlirIgemm::PerformanceConvMlirIgemm(bool spare)
    : PerformanceConvMlirIgemm::PerformanceConvMlirIgemm(64, 32, 32, 4, 2, 2, spare)
{
    if(spare)
        SetMlirHeuristicInitRequest();
}

bool PerformanceConvMlirIgemm::operator==(const PerformanceConvMlirIgemm& other) const
{
    // clang-format off
    return BlockSize == other.BlockSize
        && GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmMPerThread == other.GemmMPerThread
        && GemmNPerThread == other.GemmNPerThread;
    // clang-format on
}

bool PerformanceConvMlirIgemm::IsValid(const ExecutionContext& ctx,
                                       const ProblemDescription& problem) const
{
#if MIOPEN_USE_MLIR
    if(*this == MlirHeuristicInitRequest())
        return true;

    int kernel_count = MiirGetKernelCount(mlir::ConstructBuildOptions(ctx, problem, *this, false));
    bool isValid     = false;
    for(int kernel_id = 0; kernel_id < kernel_count; ++kernel_id)
    {
        isValid = MiirIsConfigApplicable(
            mlir::ConstructBuildOptions(ctx, problem, *this, false, kernel_id));
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

bool PerformanceConvMlirIgemm::SetNextValue(const ProblemDescription&)
{
    if(use_spare_set)
        return false;

    do
    {
        if(!NextTwoPower<64, 256>(BlockSize))
            break;
        if(!NextTwoPower<32, 128>(GemmMPerBlock))
            break;
        if(!NextTwoPower<32, 128>(GemmNPerBlock))
            break;
        if(!NextTwoPower<4, 16>(GemmKPerBlock))
            break;
        if(!NextTwoPower<2, 4>(GemmMPerThread))
            break;
        if(!NextTwoPower<2, 4>(GemmNPerThread))
            break;

        return false;
    } while(false);

    return true;
}

PerformanceConvMlirIgemm
ConvMlirIgemmFwd::GetDefaultPerformanceConfig(const ExecutionContext&,
                                              const ProblemDescription&) const
{
    return PerformanceConvMlirIgemm::MlirHeuristicInitRequest();
}

bool ConvMlirIgemmFwd::IsValidPerformanceConfig(const ExecutionContext& ctx,
                                                const ProblemDescription& problem,
                                                const PerformanceConvMlirIgemm& config) const
{
    MIOPEN_LOG_I("");
    return config.IsValid(ctx, problem);
}

PerformanceConvMlirIgemm ConvMlirIgemmFwd::Search(const ExecutionContext& ctx,
                                                  const ProblemDescription& problem,
                                                  const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvMlirIgemmFwd::IsApplicable(const ExecutionContext& ctx,
                                    const ProblemDescription& problem) const
{
#if MIOPEN_USE_MLIR
    if(env::disabled(MIOPEN_DEBUG_CONV_MLIR_IGEMM_FWD))
        return false;
    if(problem.GetConv().attribute.deterministic)
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
    // Note: ConvMlirIgemmFwd can run on a machine with xdlops support, however, it is
    // guaranteed to be slower than its xdlops alternative, therefore disabling it to
    // save compilation overhead
    if(IsXdlopsSupport(ctx))
        return false;
    // Refer to https://github.com/ROCm/llvm-project-private/issues/389
    const auto device_name = ctx.GetStream().GetDeviceName();
    if(StartsWith(device_name, "gfx900"))
        return false;

    return MiirIsConfigApplicable(mlir::ConstructBuildOptions(ctx, problem, false));
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution ConvMlirIgemmFwd::GetSolution(const ExecutionContext& ctx,
                                           const ProblemDescription& problem,
                                           const PerformanceConvMlirIgemm& config) const
{
#if MIOPEN_USE_MLIR
    ConvSolution result;
    KernelInfo construction_parameters;

    construction_parameters.kernel_name  = mlir::GetKernelName(problem, false);
    construction_parameters.kernel_file  = construction_parameters.kernel_name + ".mlir";
    construction_parameters.comp_options = mlir::ConstructBuildOptions(ctx, problem, config, false);

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
