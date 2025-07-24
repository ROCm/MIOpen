/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#pragma once

#include <algorithm>
#include <tuple>
#include <vector>

#include <miopen/conv/problem_description.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/solver/static_ck_common.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM)

namespace miopen {
namespace solver {
namespace static_ck {

static inline bool IsIndexRangeLargeEnough(const miopen::conv::ProblemDescription& problem)
{
    // composable kernel use int32_t for memory offset, which covers 2GB of memory maximum
    const std::size_t max_index_range = std::size_t(2) * 1024 * 1024 * 1024;

    return problem.GetInSize() < max_index_range && problem.GetWeightsSize() < max_index_range &&
           problem.GetOutSize() < max_index_range;
}

static inline bool IsValidBlockwiseGemmXdlops(const miopen::conv::ProblemDescription& problem,
                                              const int GemmMPerBlock,
                                              const int GemmNPerBlock,
                                              const int GemmKPerBlock,
                                              const int GemmMPerWave,
                                              const int GemmNPerWave,
                                              const int GemmKPack)
{
#if WORKAROUND_SWDEV_251757
    if(problem.IsFp32() && GemmKPerBlock == 1 && GemmKPack == 8)
        return false;
#endif

    // check k
    if(problem.IsFp16() && GemmKPack % 4 != 0)
        return false;
    if(problem.IsBfp16() && GemmKPack % 2 != 0)
        return false;

    // check M, N and K
    const std::vector<std::tuple<int, int, int>> validWaveGemmSize = {
        // std::make_tuple(128, 128, 1),
        std::make_tuple(128, 64, 1),
        // std::make_tuple(128, 32, 1),
        // std::make_tuple(128, 16, 1),
        std::make_tuple(64, 128, 1),
        std::make_tuple(64, 64, 1),
        std::make_tuple(64, 32, 1),
        std::make_tuple(64, 16, 1),
        // std::make_tuple(32, 128, 1),
        std::make_tuple(32, 64, 1),
        std::make_tuple(32, 32, 2),
        // std::make_tuple(16, 128, 1),
        std::make_tuple(16, 64, 1),
        std::make_tuple(16, 16, 4),
        // std::make_tuple(8, 128, 1),
        std::make_tuple(8, 64, 1),
        // std::make_tuple(4, 128, 1),
        std::make_tuple(4, 64, 1)};

    // NOLINTBEGIN(bugprone-assignment-in-if-condition)
    if(!std::any_of(validWaveGemmSize.cbegin(),
                    validWaveGemmSize.cend(),
                    [GemmMPerWave, GemmNPerWave, GemmKPerBlock](const auto it) noexcept -> bool {
                        int validMPerWave, validNPerWave, validKPerWave;
                        std::tie(validMPerWave, validNPerWave, validKPerWave) = it;
                        return (GemmMPerWave == validMPerWave) && (GemmNPerWave == validNPerWave) &&
                               (GemmKPerBlock % validKPerWave == 0);
                    }))
        return false;
    // NOLINTEND(bugprone-assignment-in-if-condition)

    const auto WaveSize = 64;
    const auto BlockSize =
        (GemmNPerBlock * GemmMPerBlock) / (GemmMPerWave * GemmNPerWave) * WaveSize;

    if(BlockSize < 64 || BlockSize > 256)
        return false;

    return (GemmMPerBlock % GemmMPerWave) == 0 && (GemmNPerBlock % GemmNPerWave) == 0;
}

static inline bool
IsValidGridGemmXdlops(const std::size_t GemmM, const std::size_t GemmN, const std::size_t GemmK)
{
    // unsupported xdlops-gemm
    if(GemmM % 16 != 0 && GemmN % 64 != 0)
        return false;

    const auto WaveSize = 64;

    return (GemmM * GemmN) % 256 == 0 && (GemmK * GemmM) % WaveSize == 0 &&
           (GemmK * GemmN) % WaveSize == 0 && GemmN % 16 == 0 && GemmM % 4 == 0 && GemmK % 4 == 0;
}

///\todo remove
static inline bool IsApplicableXdlops(const ExecutionContext& ctx,
                                      const miopen::conv::ProblemDescription& problem)
{
    if(!IsXdlopsSupport(ctx))
        return false;

    const std::size_t n  = ProblemInterpreter::GetBatchN(problem);
    const std::size_t k  = ProblemInterpreter::GetOutputChannelK(problem) / problem.GetGroupCount();
    const std::size_t c  = ProblemInterpreter::GetInputChannelC(problem) / problem.GetGroupCount();
    const std::size_t y  = ProblemInterpreter::GetFilterHeightY(problem);
    const std::size_t x  = ProblemInterpreter::GetFilterWidthX(problem);
    const std::size_t ho = ProblemInterpreter::GetOutputHeightHo(problem);
    const std::size_t wo = ProblemInterpreter::GetOutputWidthWo(problem);

    std::size_t GemmM, GemmN, GemmK;
    // forward
    if(problem.IsDirectionForward())
    {
        // TBD/ Since bfp16/fp16 fwd kernel extracts epack from c*y*x,
        //      one could relax the following restriction for bfp16/fp16,
        //      allowing c=1 when y*x=epack.
        if(c % GetEPackLength(ctx, problem, true) != 0)
            return false;
        const auto nonVectorizedC = c / GetEPackLength(ctx, problem, true);
        GemmM                     = k;
        GemmN                     = static_cast<std::size_t>(n) * ho * wo;
        GemmK                     = static_cast<std::size_t>(nonVectorizedC) * y * x;
    }
    // backwardData
    else if(problem.IsDirectionBackwardData())
    {
        if(k % GetEPackLength(ctx, problem, true) != 0)
            return false;
        const auto nonVectorizedK = k / GetEPackLength(ctx, problem, true);
        GemmM                     = static_cast<std::size_t>(c) * y * x;
        GemmN                     = static_cast<std::size_t>(n) * ho * wo;
        GemmK                     = nonVectorizedK;
    }
    // backwardWeights
    else
    {
        if(n % GetEPackLength(ctx, problem, true) != 0)
            return false;
        const auto nonVectorizedN = n / GetEPackLength(ctx, problem, true);
        GemmM                     = k;
        GemmN                     = static_cast<std::size_t>(c) * y * x;
        GemmK                     = static_cast<std::size_t>(nonVectorizedN) * ho * wo;
    }

    return IsValidGridGemmXdlops(GemmM, GemmN, GemmK);
}

///\todo remove
template <class PerformanceImplicitGemm_t>
inline static auto GetPerformanceConfigBase(const ExecutionContext& ctx,
                                            const miopen::conv::ProblemDescription& problem)
{
    PerformanceImplicitGemm_t pp;
    pp.HeuristicInit(ctx, problem);
    MIOPEN_LOG_I(pp.ToString());
    return pp;
}

static inline bool use_amd_inline_asm(const ExecutionContext& ctx,
                                      const miopen::conv::ProblemDescription& problem)
{

    if(StartsWith(ctx.GetStream().GetDeviceName(), "gfx8"))
        return false;

    // disable fp16 inline asm for <= gfx900
    const auto device_name = ctx.GetStream().GetDeviceName();
    if(!(StartsWith(device_name, "gfx906") || StartsWith(device_name, "gfx908")) &&
       problem.IsFp16())
        return false;

    return !env::disabled(MIOPEN_DEBUG_IMPLICIT_GEMM_NON_XDLOPS_INLINE_ASM);
}

} // namespace static_ck
} // namespace solver
} // namespace miopen
