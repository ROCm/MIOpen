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

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_MLIR_IGEMM_BWD)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvMlirIgemmBwd::IsApplicable(const ExecutionContext& ctx,
                                    const ProblemDescription& problem) const
{
#if MIOPEN_USE_MLIR
    if(env::disabled(MIOPEN_DEBUG_CONV_MLIR_IGEMM_BWD))
        return false;
    if(problem.GetConv().attribute.deterministic)
        return false;
    if(!problem.IsDirectionBackwardData())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;
    if(problem.IsTensorsCasted() || problem.IsFp8() || problem.IsBfp8())
        return false;
    // Note: ConvMlirIgemmBwd can run on a machine with xdlops support, however, it is
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

PerformanceConvMlirIgemm
ConvMlirIgemmBwd::GetDefaultPerformanceConfig(const ExecutionContext&,
                                              const ProblemDescription&) const
{
    return PerformanceConvMlirIgemm::MlirHeuristicInitRequest();
}

bool ConvMlirIgemmBwd::IsValidPerformanceConfig(const ExecutionContext& ctx,
                                                const ProblemDescription& problem,
                                                const PerformanceConvMlirIgemm& config) const
{
    MIOPEN_LOG_I("");
    return config.IsValid(ctx, problem);
}

PerformanceConvMlirIgemm ConvMlirIgemmBwd::Search(const ExecutionContext& ctx,
                                                  const ProblemDescription& problem,
                                                  const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution ConvMlirIgemmBwd::GetSolution(const ExecutionContext& ctx,
                                           const ProblemDescription& problem,
                                           const PerformanceConvMlirIgemm& config) const
{
#if MIOPEN_USE_MLIR
    ConvSolution result;
    int kernel_count = MiirGetKernelCount(mlir::ConstructBuildOptions(ctx, problem, config, false));

    for(int kernel_id = 0; kernel_id < kernel_count; ++kernel_id)
    {
        KernelInfo construction_parameters;

        construction_parameters.kernel_name = mlir::GetKernelName(problem, false, kernel_id);
        construction_parameters.kernel_file = construction_parameters.kernel_name + ".mlir";
        construction_parameters.comp_options =
            mlir::ConstructBuildOptions(ctx, problem, config, false, kernel_id);

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

    result.invoker_factory = miopen::conv::MakeMlirBwdInvokerFactory(problem);
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
