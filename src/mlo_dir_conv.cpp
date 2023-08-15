/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/config.h>
#include <miopen/convolution.hpp>
#include <miopen/db.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver.hpp>
#include <miopen/readonlyramdb.hpp>
#include <miopen/datatype.hpp>
#include <miopen/version.h>
#include <miopen/stringutils.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/any_solver.hpp>

#include <cmath>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <unordered_map>

#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif

// Only select the first applicable igemm solver due to long compilation time
// (JIRA SWDEV-227826)
/// \todo enable all applicable solvers of igemm after fixing slow compilation
#define WORKAROUND_SWDEV_227826 0

#if WORKAROUND_SWDEV_227826
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS)
#endif

miopen::PerformanceDb miopen::GetDb(const miopen::ExecutionContext& ctx)
{
    return {ctx.GetPerfDbPath(), ctx.GetUserPerfDbPath()};
}

static auto GetGemmSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::GemmFwd1x1_0_1,
                                           miopen::solver::GemmFwd1x1_0_1_int8,
                                           miopen::solver::GemmFwd1x1_0_2,
                                           miopen::solver::GemmFwdRest,

                                           miopen::solver::GemmBwd1x1_stride1,
                                           miopen::solver::GemmBwd1x1_stride2,
                                           miopen::solver::GemmBwdRest,

                                           miopen::solver::GemmWrw1x1_stride1,
                                           miopen::solver::GemmWrwUniversal>{};
}

static auto GetDirectSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvAsm3x3U,
                                           miopen::solver::ConvAsm1x1U,
                                           miopen::solver::ConvAsm1x1UV2,
                                           miopen::solver::ConvAsm5x10u2v2f1,
                                           miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                                           miopen::solver::ConvAsm5x10u2v2b1,
                                           miopen::solver::ConvOclDirectFwd11x11,
                                           miopen::solver::ConvOclDirectFwdGen,
                                           miopen::solver::ConvOclDirectFwd1x1,
                                           miopen::solver::ConvOclDirectFwd,
                                           miopen::solver::ConvDirectNaiveConvFwd,
                                           miopen::solver::ConvDirectNaiveConvBwd,
                                           miopen::solver::ConvDirectNaiveConvWrw>{};
}

static auto GetImplicitGemmSolvers()
{
    return miopen::solver::SolverContainer<
        miopen::solver::ConvHipImplicitGemmForwardV4R5Xdlops,
        miopen::solver::ConvHipImplicitGemmForwardV4R4Xdlops,
        miopen::solver::ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm,
        miopen::solver::ConvHipImplicitGemmBwdDataV4R1Xdlops,
        miopen::solver::ConvHipImplicitGemmBwdDataV1R1Xdlops,
        miopen::solver::ConvHipImplicitGemmV4R1Fwd,
        miopen::solver::ConvHipImplicitGemmV4R4Fwd,
        miopen::solver::ConvMlirIgemmFwdXdlops,
        miopen::solver::ConvMlirIgemmFwd,
        miopen::solver::ConvMlirIgemmBwdXdlops,
        miopen::solver::ConvMlirIgemmBwd,
        miopen::solver::ConvHipImplicitGemmBwdDataV1R1,
        miopen::solver::ConvHipImplicitGemmBwdDataV4R1,
        miopen::solver::ConvAsmImplicitGemmV4R1DynamicFwd_1x1,
        miopen::solver::ConvAsmImplicitGemmV4R1DynamicFwd,
        miopen::solver::ConvAsmImplicitGemmV4R1DynamicBwd,
        miopen::solver::ConvAsmImplicitGemmGTCDynamicFwdXdlops,
        miopen::solver::ConvAsmImplicitGemmGTCDynamicBwdXdlops,
        miopen::solver::ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC,
        miopen::solver::ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC,
        miopen::solver::ConvCkIgemmFwdV6r1DlopsNchw,
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        miopen::solver::ConvHipImplicitGemmFwdXdlops,
        miopen::solver::ConvHipImplicitGemmBwdXdlops,
        miopen::solver::ConvHipImplicitGemmGroupFwdXdlops,
        miopen::solver::ConvHipImplicitGemm3DGroupFwdXdlops,
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        miopen::solver::ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC>{};
}

static auto GetWindogradSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvBinWinograd3x3U,
                                           miopen::solver::ConvBinWinoRxS<3, 2>,
                                           miopen::solver::ConvBinWinoRxS<2, 3>,
                                           miopen::solver::ConvBinWinogradRxSf2x3g1,
                                           miopen::solver::ConvBinWinogradRxS,
                                           miopen::solver::ConvMPBidirectWinograd<3, 3>,
                                           miopen::solver::ConvMPBidirectWinograd<4, 3>,
                                           miopen::solver::ConvMPBidirectWinograd<5, 3>,
                                           miopen::solver::ConvMPBidirectWinograd<6, 3>,
                                           miopen::solver::ConvMPBidirectWinograd_xdlops<2, 3>,
                                           miopen::solver::ConvMPBidirectWinograd_xdlops<3, 3>,
                                           miopen::solver::ConvMPBidirectWinograd_xdlops<4, 3>,
                                           miopen::solver::ConvMPBidirectWinograd_xdlops<5, 3>,
                                           miopen::solver::ConvMPBidirectWinograd_xdlops<6, 3>>{};
}

static auto GetImplicitGemmWrWSolvers()
{
    return miopen::solver::SolverContainer<
        miopen::solver::ConvHipImplicitGemmWrwV4R4Xdlops,
        miopen::solver::ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm,
        miopen::solver::ConvHipImplicitGemmV4R1WrW,
        miopen::solver::ConvHipImplicitGemmV4R4WrW,
        miopen::solver::ConvAsmImplicitGemmV4R1DynamicWrw,
        miopen::solver::ConvMlirIgemmWrWXdlops,
        miopen::solver::ConvMlirIgemmWrW,
        miopen::solver::ConvAsmImplicitGemmGTCDynamicWrwXdlops,
        miopen::solver::ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC>{};
}

static auto GetWindogradWrWSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvBinWinogradRxS,
                                           miopen::solver::ConvBinWinoRxS<3, 2>,
                                           miopen::solver::ConvBinWinoRxS<2, 3>,
                                           miopen::solver::ConvBinWinogradRxSf2x3g1,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<3, 2>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<3, 3>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<3, 4>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<3, 5>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<3, 6>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<7, 2>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<7, 3>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<7, 3, 1, 1>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<7, 2, 1, 1>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<1, 1, 7, 2>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<1, 1, 7, 3>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<5, 3>,
                                           miopen::solver::ConvWinograd3x3MultipassWrW<5, 4>>{};
}

static auto GetBwdWrW2DSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvAsmBwdWrW1x1,
                                           miopen::solver::ConvAsmBwdWrW3x3,
                                           miopen::solver::ConvOclBwdWrW2<1>,
                                           miopen::solver::ConvOclBwdWrW2<2>,
                                           miopen::solver::ConvOclBwdWrW2<4>,
                                           miopen::solver::ConvOclBwdWrW2<8>,
                                           miopen::solver::ConvOclBwdWrW2<16>,
                                           miopen::solver::ConvOclBwdWrW2NonTunable,
                                           miopen::solver::ConvOclBwdWrW53,
                                           miopen::solver::ConvOclBwdWrW1x1,
                                           miopen::solver::ConvDirectNaiveConvFwd,
                                           miopen::solver::ConvDirectNaiveConvBwd,
                                           miopen::solver::ConvDirectNaiveConvWrw>{};
}

static auto GetFFTSolvers() { return miopen::solver::SolverContainer<miopen::solver::fft>{}; }

bool IsGemmAplicable(const miopen::ConvolutionContext& ctx,
                     const miopen::ProblemDescription& problem)
{
    return GetGemmSolvers().IsAnySolverApplicable(ctx, problem);
}

std::vector<miopen::solver::ConvSolution>
FindAllGemmSolutions(const miopen::ConvolutionContext& ctx,
                     const miopen::ProblemDescription& problem,
                     const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetGemmSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllGemmWorkspaceSize(const miopen::ConvolutionContext& ctx,
                     const miopen::ProblemDescription& problem)
{
    return GetGemmSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<miopen::solver::ConvSolution>
FindAllDirectSolutions(const miopen::ConvolutionContext& ctx,
                       const miopen::ProblemDescription& problem,
                       const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetDirectSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllDirectForwardBackwardDataWorkspaceSize(const miopen::ConvolutionContext& ctx,
                                          const miopen::ProblemDescription& problem)
{
    return GetDirectSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<std::pair<std::string, size_t>>
FindAllWinogradWorkspaceSizes(const miopen::ConvolutionContext& ctx,
                              const miopen::ProblemDescription& problem)
{
    return GetWindogradSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<std::pair<std::string, size_t>>
FindWinogradWrWWorkspaceSizes(const miopen::ConvolutionContext& ctx,
                              const miopen::ProblemDescription& problem)
{
    return GetWindogradWrWSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<std::pair<std::string, size_t>>
FindAllImplicitGemmWorkspaceSizes(const miopen::ConvolutionContext& ctx,
                                  const miopen::ProblemDescription& problem)
{
#if WORKAROUND_SWDEV_227826
    if(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS{}))
        return GetImplicitGemmSolvers().GetWorkspaceSizes(ctx, problem);
    else
        return GetImplicitGemmSolvers().GetWorkspaceSizes(ctx, problem, 1);
#else
    return GetImplicitGemmSolvers().GetWorkspaceSizes(ctx, problem);
#endif
}

std::vector<miopen::solver::ConvSolution>
FindAllImplicitGemmSolutions(const miopen::ConvolutionContext& ctx,
                             const miopen::ProblemDescription& problem,
                             const miopen::AnyInvokeParams& invoke_ctx)
{
#if WORKAROUND_SWDEV_227826
    if(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS{}))
        return GetImplicitGemmSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
    else
        return GetImplicitGemmSolvers().SearchForAllSolutions(
            ctx, problem, GetDb(ctx), invoke_ctx, 1);
#else
    return GetImplicitGemmSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
#endif
}

std::vector<miopen::solver::ConvSolution>
FindAllWinogradSolutions(const miopen::ConvolutionContext& ctx,
                         const miopen::ProblemDescription& problem,
                         const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetWindogradSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<miopen::solver::ConvSolution>
FindWinogradWrWAllSolutions(const miopen::ConvolutionContext& ctx,
                            const miopen::ProblemDescription& problem,
                            const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetWindogradWrWSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllDirectBwdWrW2DWorkspaceSize(const miopen::ConvolutionContext& ctx,
                               const miopen::ProblemDescription& problem)
{
    return GetBwdWrW2DSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<std::pair<std::string, size_t>>
FindImplicitGemmWrWWorkspaceSizes(const miopen::ConvolutionContext& ctx,
                                  const miopen::ProblemDescription& problem)
{
#if WORKAROUND_SWDEV_227826
    if(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS{}))
        return GetImplicitGemmWrWSolvers().GetWorkspaceSizes(ctx, problem);
    else
        return GetImplicitGemmWrWSolvers().GetWorkspaceSizes(ctx, problem, 1);
#else
    return GetImplicitGemmWrWSolvers().GetWorkspaceSizes(ctx, problem);
#endif
}

std::vector<miopen::solver::ConvSolution>
FindImplicitGemmWrWAllSolutions(const miopen::ConvolutionContext& ctx,
                                const miopen::ProblemDescription& problem,
                                const miopen::AnyInvokeParams& invoke_ctx)
{
#if WORKAROUND_SWDEV_227826
    if(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS{}))
        return GetImplicitGemmWrWSolvers().SearchForAllSolutions(
            ctx, problem, GetDb(ctx), invoke_ctx);
    else
        return GetImplicitGemmWrWSolvers().SearchForAllSolutions(
            ctx, problem, GetDb(ctx), invoke_ctx, 1);
#else
    return GetImplicitGemmWrWSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
#endif
}

std::vector<miopen::solver::ConvSolution>
FindAllBwdWrW2DSolutions(const miopen::ConvolutionContext& ctx,
                         const miopen::ProblemDescription& problem,
                         const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetBwdWrW2DSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<miopen::solver::ConvSolution>
FindAllFFTSolutions(const miopen::ConvolutionContext& ctx,
                    const miopen::ProblemDescription& problem,
                    const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetFFTSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllFFTForwardBackwardDataWorkspaceSize(const miopen::ConvolutionContext& ctx,
                                       const miopen::ProblemDescription& problem)
{
    return GetFFTSolvers().GetWorkspaceSizes(ctx, problem);
}

void mlo_construct_activ_lrn_pooling_common::setupFloats()
{
    if(_problem.GetInDataType() == miopenFloat && _problem.GetOutDataType() == miopenFloat)
    {
        _ctx.general_compile_options += " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";
    }
    else if(_problem.GetInDataType() == miopenHalf && _problem.GetOutDataType() == miopenHalf)
    {
        _ctx.general_compile_options += " -DMIOPEN_USE_FP32=0 -DMIOPEN_USE_FP16=1";
    }
    else
    {
        MIOPEN_LOG_W("Unsupported data types configuration: "
                     << miopen::GetDataTypeName(_problem.GetInDataType()) << "x"
                     << miopen::GetDataTypeName(_problem.GetOutDataType()));
    }
}
