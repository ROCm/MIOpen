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
#include <miopen/solver.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/mlir_common.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_MLIR_IGEMM_FWD)

namespace miopen {
namespace solver {

const PerformanceConvMlirIgemm& PerformanceConvMlirIgemm::MlirHeuristicInitRequest()
{
    static const PerformanceConvMlirIgemm p =
        PerformanceConvMlirIgemm(-2, -2, -2, -2, -2, -2, false);
    return p;
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
{
    // always search full space, no matter if use_spare_set or not
    BlockSize = 64;

    GemmMPerBlock = 32;
    GemmNPerBlock = 32;
    GemmKPerBlock = 4;

    GemmMPerThread = 2;
    GemmNPerThread = 2;

    use_spare_set = spare;
}

bool PerformanceConvMlirIgemm::operator==(const PerformanceConvMlirIgemm& other) const
{
    // clang-format off
    return BlockSize == other.BlockSize
        && GemmMPerBlock == other.GemmMPerBlock
        && GemmNPerBlock == other.GemmNPerBlock
        && GemmKPerBlock == other.GemmKPerBlock
        && GemmMPerThread == other.GemmMPerThread
        && GemmNPerThread == other.GemmNPerThread
        && use_spare_set == other.use_spare_set;
    // clang-format on
}

bool PerformanceConvMlirIgemm::IsValid(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    if(*this == MlirHeuristicInitRequest())
        return true;

    return MiirIsConfigApplicable(mlir::ConstructBuildOptions(ctx, *this, false));
#else
    std::ignore = ctx;
    return false;
#endif
}

bool PerformanceConvMlirIgemm::SetNextValue(const ConvolutionContext& /*config*/)
{
    if(*this == MlirHeuristicInitRequest())
        MIOPEN_THROW("Should not iterate from the heuristic value");

    // always search full space, no matter if use_spare_set or not
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

std::string PerformanceConvMlirIgemm::ToString() const
{
    std::ostringstream ss;
    Serialize(ss);
    return ss.str();
}

PerformanceConvMlirIgemm ConvMlirIgemmFwd::GetPerformanceConfig(const ConvolutionContext& ctx) const
{
    std::ignore = ctx;
    return PerformanceConvMlirIgemm::MlirHeuristicInitRequest();
}

bool ConvMlirIgemmFwd::IsValidPerformanceConfig(const ConvolutionContext& ctx,
                                                const PerformanceConvMlirIgemm& config) const
{
    MIOPEN_LOG_I("");
    return config.IsValid(ctx);
}

PerformanceConvMlirIgemm ConvMlirIgemmFwd::Search(const ConvolutionContext& context,
                                                  const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, context, invoke_ctx);
}

bool ConvMlirIgemmFwd::IsApplicable(const ConvolutionContext& ctx) const
{
#if MIOPEN_USE_MLIR
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_MLIR_IGEMM_FWD{}))
        return false;
    if(!ctx.direction.IsForward())
        return false;
    if(!IsComposableKernelSupportedHardware(ctx))
        return false;

    return MiirIsConfigApplicable(mlir::ConstructBuildOptions(ctx, false));
#else
    std::ignore = ctx;
    return false;
#endif
}

ConvSolution ConvMlirIgemmFwd::GetSolution(const ConvolutionContext& ctx,
                                           const PerformanceConvMlirIgemm& config,
                                           bool) const
{
#if MIOPEN_USE_MLIR
    ConvSolution result;
    KernelInfo construction_parameters;

    construction_parameters.kernel_name  = mlir::GetKernelName(ctx, false);
    construction_parameters.kernel_file  = construction_parameters.kernel_name + ".mlir";
    construction_parameters.comp_options = mlir::ConstructBuildOptions(ctx, config, false);

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
