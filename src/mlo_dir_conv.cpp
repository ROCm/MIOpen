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
#include <miopen/conv/solvers.hpp>
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

miopen::PerformanceDb miopen::GetDb(const miopen::ExecutionContext& ctx)
{
    return {DbKinds::PerfDb, ctx.GetPerfDbPath(), ctx.GetUserPerfDbPath()};
}

static auto GetGemmSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::conv::GemmFwd1x1_0_1,
                                           miopen::solver::conv::GemmFwd1x1_0_1_int8,
                                           miopen::solver::conv::GemmFwd1x1_0_2,
                                           miopen::solver::conv::GemmFwdRest,

                                           miopen::solver::conv::GemmBwd1x1_stride1,
                                           miopen::solver::conv::GemmBwd1x1_stride2,
                                           miopen::solver::conv::GemmBwdRest,

                                           miopen::solver::conv::GemmWrw1x1_stride1,
                                           miopen::solver::conv::GemmWrwUniversal>{};
}

static auto GetDirectSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::conv::ConvAsm3x3U,
                                           miopen::solver::conv::ConvAsm1x1U,
                                           miopen::solver::conv::ConvAsm1x1UV2,
                                           miopen::solver::conv::ConvAsm5x10u2v2f1,
                                           miopen::solver::conv::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
                                           miopen::solver::conv::ConvAsm5x10u2v2b1,
                                           miopen::solver::conv::ConvOclDirectFwd11x11,
                                           miopen::solver::conv::ConvOclDirectFwdGen,
                                           miopen::solver::conv::ConvOclDirectFwd1x1,
                                           miopen::solver::conv::ConvOclDirectFwd,
                                           miopen::solver::conv::ConvDirectNaiveConvFwd,
                                           miopen::solver::conv::ConvDirectNaiveConvBwd,
                                           miopen::solver::conv::ConvDirectNaiveConvWrw>{};
}

static auto GetImplicitGemmSolvers()
{
    return miopen::solver::SolverContainer<
        miopen::solver::conv::ConvHipImplicitGemmForwardV4R5Xdlops,
        miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops,
        miopen::solver::conv::ConvHipImplicitGemmForwardV4R4Xdlops_Padded_Gemm,
        miopen::solver::conv::ConvHipImplicitGemmBwdDataV4R1Xdlops,
        miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1Xdlops,
        miopen::solver::conv::ConvHipImplicitGemmV4R1Fwd,
        miopen::solver::conv::ConvHipImplicitGemmV4R4Fwd,
        miopen::solver::conv::ConvMlirIgemmFwdXdlops,
        miopen::solver::conv::ConvMlirIgemmFwd,
        miopen::solver::conv::ConvMlirIgemmBwdXdlops,
        miopen::solver::conv::ConvMlirIgemmBwd,
        miopen::solver::conv::ConvHipImplicitGemmBwdDataV1R1,
        miopen::solver::conv::ConvHipImplicitGemmBwdDataV4R1,
        miopen::solver::conv::ConvAsmImplicitGemmV4R1DynamicFwd_1x1,
        miopen::solver::conv::ConvAsmImplicitGemmV4R1DynamicFwd,
        miopen::solver::conv::ConvAsmImplicitGemmV4R1DynamicBwd,
        miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicFwdXdlops,
        miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicBwdXdlops,
        miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC,
        miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC,
        miopen::solver::conv::ConvCkIgemmFwdV6r1DlopsNchw,
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        miopen::solver::conv::ConvHipImplicitGemmFwdXdlops,
        miopen::solver::conv::ConvHipImplicitGemmBwdXdlops,
        miopen::solver::conv::ConvHipImplicitGemmGroupFwdXdlops,
        miopen::solver::conv::ConvHipImplicitGemmGroupBwdXdlops,
        miopen::solver::conv::ConvHipImplicitGemm3DGroupFwdXdlops,
        miopen::solver::conv::ConvHipImplicitGemm3DGroupBwdXdlops,
        miopen::solver::conv::ConvHipImplicitGemmF16F8F16FwdXdlops,
        miopen::solver::conv::ConvHipImplicitGemmF16F8F16BwdXdlops,
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicFwdDlopsNCHWC>{};
}

static auto GetWindogradSolvers()
{
    return miopen::solver::SolverContainer<
        miopen::solver::conv::ConvBinWinograd3x3U,
        miopen::solver::conv::ConvBinWinoRxS<3, 2>,
        miopen::solver::conv::ConvBinWinoRxS<2, 3>,
        miopen::solver::conv::ConvBinWinogradRxSf2x3g1,
        miopen::solver::conv::ConvBinWinogradRxS,
        miopen::solver::conv::ConvMPBidirectWinograd<3, 3>,
        miopen::solver::conv::ConvMPBidirectWinograd<4, 3>,
        miopen::solver::conv::ConvMPBidirectWinograd<5, 3>,
        miopen::solver::conv::ConvMPBidirectWinograd<6, 3>,
        miopen::solver::conv::ConvMPBidirectWinograd_xdlops<2, 3>,
        miopen::solver::conv::ConvMPBidirectWinograd_xdlops<3, 3>,
        miopen::solver::conv::ConvMPBidirectWinograd_xdlops<4, 3>,
        miopen::solver::conv::ConvMPBidirectWinograd_xdlops<5, 3>,
        miopen::solver::conv::ConvMPBidirectWinograd_xdlops<6, 3>,
        miopen::solver::conv::ConvWinoFuryRxS<2, 3>>{};
}

static auto GetImplicitGemmWrWSolvers()
{
    return miopen::solver::SolverContainer<
        miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops,
        miopen::solver::conv::ConvHipImplicitGemmWrwV4R4Xdlops_Padded_Gemm,
        miopen::solver::conv::ConvHipImplicitGemmV4R1WrW,
        miopen::solver::conv::ConvHipImplicitGemmV4R4WrW,
        miopen::solver::conv::ConvAsmImplicitGemmV4R1DynamicWrw,
        miopen::solver::conv::ConvMlirIgemmWrWXdlops,
        miopen::solver::conv::ConvMlirIgemmWrW,
        miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicWrwXdlops,
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        miopen::solver::conv::ConvHipImplicitGemmGroupWrwXdlops,
        miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops,
        miopen::solver::conv::ConvHipImplicitGemmF16F8F16WrwXdlops,
#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
        miopen::solver::conv::ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC>{};
}

static auto GetWindogradWrWSolvers()
{
    return miopen::solver::SolverContainer<
        miopen::solver::conv::ConvBinWinogradRxS,
        miopen::solver::conv::ConvBinWinoRxS<3, 2>,
        miopen::solver::conv::ConvBinWinoRxS<2, 3>,
        miopen::solver::conv::ConvBinWinogradRxSf2x3g1,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<3, 2>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<3, 3>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<3, 4>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<3, 5>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<3, 6>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<7, 2>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<7, 3>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<7, 3, 1, 1>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<7, 2, 1, 1>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<1, 1, 7, 2>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<1, 1, 7, 3>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<5, 3>,
        miopen::solver::conv::ConvWinograd3x3MultipassWrW<5, 4>,
        miopen::solver::conv::ConvWinoFuryRxS<2, 3>>{};
}

static auto GetBwdWrW2DSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::conv::ConvAsmBwdWrW1x1,
                                           miopen::solver::conv::ConvAsmBwdWrW3x3,
                                           miopen::solver::conv::ConvOclBwdWrW2<1>,
                                           miopen::solver::conv::ConvOclBwdWrW2<2>,
                                           miopen::solver::conv::ConvOclBwdWrW2<4>,
                                           miopen::solver::conv::ConvOclBwdWrW2<8>,
                                           miopen::solver::conv::ConvOclBwdWrW2<16>,
                                           miopen::solver::conv::ConvOclBwdWrW2NonTunable,
                                           miopen::solver::conv::ConvOclBwdWrW53,
                                           miopen::solver::conv::ConvOclBwdWrW1x1,
                                           miopen::solver::conv::ConvDirectNaiveConvFwd,
                                           miopen::solver::conv::ConvDirectNaiveConvBwd,
                                           miopen::solver::conv::ConvDirectNaiveConvWrw>{};
}

static auto GetFFTSolvers() { return miopen::solver::SolverContainer<miopen::solver::conv::fft>{}; }

std::vector<miopen::solver::ConvSolution>
FindAllGemmSolutions(const miopen::ExecutionContext& ctx,
                     const miopen::conv::ProblemDescription& problem,
                     const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetGemmSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllGemmWorkspaceSize(const miopen::ExecutionContext& ctx,
                     const miopen::conv::ProblemDescription& problem)
{
    return GetGemmSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<miopen::solver::ConvSolution>
FindAllDirectSolutions(const miopen::ExecutionContext& ctx,
                       const miopen::conv::ProblemDescription& problem,
                       const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetDirectSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllDirectForwardBackwardDataWorkspaceSize(const miopen::ExecutionContext& ctx,
                                          const miopen::conv::ProblemDescription& problem)
{
    return GetDirectSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<std::pair<std::string, size_t>>
FindAllWinogradWorkspaceSizes(const miopen::ExecutionContext& ctx,
                              const miopen::conv::ProblemDescription& problem)
{
    return GetWindogradSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<std::pair<std::string, size_t>>
FindWinogradWrWWorkspaceSizes(const miopen::ExecutionContext& ctx,
                              const miopen::conv::ProblemDescription& problem)
{
    return GetWindogradWrWSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<std::pair<std::string, size_t>>
FindAllImplicitGemmWorkspaceSizes(const miopen::ExecutionContext& ctx,
                                  const miopen::conv::ProblemDescription& problem)
{
    return GetImplicitGemmSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<miopen::solver::ConvSolution>
FindAllImplicitGemmSolutions(const miopen::ExecutionContext& ctx,
                             const miopen::conv::ProblemDescription& problem,
                             const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetImplicitGemmSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<miopen::solver::ConvSolution>
FindAllWinogradSolutions(const miopen::ExecutionContext& ctx,
                         const miopen::conv::ProblemDescription& problem,
                         const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetWindogradSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<miopen::solver::ConvSolution>
FindWinogradWrWAllSolutions(const miopen::ExecutionContext& ctx,
                            const miopen::conv::ProblemDescription& problem,
                            const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetWindogradWrWSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllDirectBwdWrW2DWorkspaceSize(const miopen::ExecutionContext& ctx,
                               const miopen::conv::ProblemDescription& problem)
{
    return GetBwdWrW2DSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<std::pair<std::string, size_t>>
FindImplicitGemmWrWWorkspaceSizes(const miopen::ExecutionContext& ctx,
                                  const miopen::conv::ProblemDescription& problem)
{
    return GetImplicitGemmWrWSolvers().GetWorkspaceSizes(ctx, problem);
}

std::vector<miopen::solver::ConvSolution>
FindImplicitGemmWrWAllSolutions(const miopen::ExecutionContext& ctx,
                                const miopen::conv::ProblemDescription& problem,
                                const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetImplicitGemmWrWSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<miopen::solver::ConvSolution>
FindAllBwdWrW2DSolutions(const miopen::ExecutionContext& ctx,
                         const miopen::conv::ProblemDescription& problem,
                         const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetBwdWrW2DSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<miopen::solver::ConvSolution>
FindAllFFTSolutions(const miopen::ExecutionContext& ctx,
                    const miopen::conv::ProblemDescription& problem,
                    const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetFFTSolvers().SearchForAllSolutions(ctx, problem, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllFFTForwardBackwardDataWorkspaceSize(const miopen::ExecutionContext& ctx,
                                       const miopen::conv::ProblemDescription& problem)
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
