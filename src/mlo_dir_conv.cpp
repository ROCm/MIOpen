/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#define MIOPEN

#include <miopen/config.h>
#include <miopen/convolution.hpp>
#include <miopen/db.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/mlo_utils.hpp>
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

#include <miopen/solver.hpp>
#if MIOPEN_ENABLE_SQLITE
#include <miopen/sqlite_db.hpp>
#endif
#include <miopen/db.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/mlo_utils.hpp>

// Only select the first applicable igemm solver due to long compilation time
// (JIRA SWDEV-227826)
/// \todo enable all applicable solvers of igemm after fixing slow compilation
#define WORKAROUND_SWDEV_227826 1

#if WORKAROUND_SWDEV_227826
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS)
#endif

miopen::PerformanceDb mlo_construct_base::GetDb() const
{
    return {db_path(), _search_params.GetUserPerfDbPath()};
}

miopen::PerformanceDb miopen::GetDb(const miopen::ExecutionContext& ctx)
{
    return {ctx.GetPerfDbPath(), ctx.GetUserPerfDbPath()};
}
miopen::solver::ConvSolution
mlo_construct_direct2D_fusion::FindSolution(const std::vector<miopen::solver::AnySolver>& solvers,
                                            const miopen::AnyInvokeParams& invoke_ctx)
{
    miopen::solver::ConvSolution solution{miopenStatusUnknownError};
    std::string solver_id;
    auto db = this->GetDb();
    for(auto& solver : solvers)
    {
        solution = solver.FindSolution(_search_params, db, invoke_ctx);
        if(solution.Succeeded() && solver.IsApplicable(_search_params))
        {
            solver_id = miopen::solver::SolverDbId(solver);
            break;
        }
    }
    if(solution.Succeeded() && solution.construction_params.empty())
    {
        MIOPEN_THROW(std::string("Internal error in solver: ") + solver_id);
    }
    return solution;
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
        miopen::solver::ConvCkIgemmFwdV6r1DlopsNchw>{};
}

static auto GetWindogradSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvBinWinograd3x3U,
                                           miopen::solver::ConvBinWinogradRxSf3x2,
                                           miopen::solver::ConvBinWinogradRxSf2x3,
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
                                           miopen::solver::ConvBinWinogradRxSf3x2,
                                           miopen::solver::ConvBinWinogradRxSf2x3,
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

bool IsGemmAplicable(const miopen::ConvolutionContext& ctx)
{
    return GetGemmSolvers().IsAnySolverApplicable(ctx);
}

std::vector<miopen::solver::ConvSolution>
FindAllGemmSolutions(const miopen::ConvolutionContext& ctx,
                     const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetGemmSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllGemmWorkspaceSize(const miopen::ConvolutionContext& ctx)
{
    return GetGemmSolvers().GetWorkspaceSize(ctx);
}

std::vector<miopen::solver::ConvSolution>
FindAllDirectSolutions(const miopen::ConvolutionContext& ctx,
                       const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetDirectSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllDirectForwardBackwardDataWorkspaceSize(const miopen::ConvolutionContext& ctx)
{
    return GetDirectSolvers().GetWorkspaceSize(ctx);
}

std::vector<std::pair<std::string, size_t>>
FindAllWinogradWorkspaceSizes(const miopen::ConvolutionContext& ctx)
{
    return GetWindogradSolvers().GetWorkspaceSize(ctx);
}

std::vector<std::pair<std::string, size_t>>
FindWinogradWrWWorkspaceSizes(const miopen::ConvolutionContext& ctx)
{
    return GetWindogradWrWSolvers().GetWorkspaceSize(ctx);
}

std::vector<std::pair<std::string, size_t>>
FindAllImplicitGemmWorkspaceSizes(const miopen::ConvolutionContext& ctx)
{
#if WORKAROUND_SWDEV_227826
    if(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS{}))
        return GetImplicitGemmSolvers().GetWorkspaceSize(ctx);
    else
        return GetImplicitGemmSolvers().GetWorkspaceSize(ctx, 1);
#else
    return GetImplicitGemmSolvers().GetWorkspaceSize(ctx);
#endif
}

std::vector<miopen::solver::ConvSolution>
FindAllImplicitGemmSolutions(const miopen::ConvolutionContext& ctx,
                             const miopen::AnyInvokeParams& invoke_ctx)
{
#if WORKAROUND_SWDEV_227826
    if(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS{}))
        return GetImplicitGemmSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
    else
        return GetImplicitGemmSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx, 1);
#else
    return GetImplicitGemmSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
#endif
}

std::vector<miopen::solver::ConvSolution>
FindAllWinogradSolutions(const miopen::ConvolutionContext& ctx,
                         const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetWindogradSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
}

std::vector<miopen::solver::ConvSolution>
FindWinogradWrWAllSolutions(const miopen::ConvolutionContext& ctx,
                            const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetWindogradWrWSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllDirectBwdWrW2DWorkspaceSize(const miopen::ConvolutionContext& ctx)
{
    return GetBwdWrW2DSolvers().GetWorkspaceSize(ctx);
}

std::vector<std::pair<std::string, size_t>>
FindImplicitGemmWrWWorkspaceSizes(const miopen::ConvolutionContext& ctx)
{
#if WORKAROUND_SWDEV_227826
    if(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS{}))
        return GetImplicitGemmWrWSolvers().GetWorkspaceSize(ctx);
    else
        return GetImplicitGemmWrWSolvers().GetWorkspaceSize(ctx, 1);
#else
    return GetImplicitGemmWrWSolvers().GetWorkspaceSize(ctx);
#endif
}

std::vector<miopen::solver::ConvSolution>
FindImplicitGemmWrWAllSolutions(const miopen::ConvolutionContext& ctx,
                                const miopen::AnyInvokeParams& invoke_ctx)
{
#if WORKAROUND_SWDEV_227826
    if(miopen::IsEnabled(MIOPEN_DEBUG_IMPLICIT_GEMM_FIND_ALL_SOLUTIONS{}))
        return GetImplicitGemmWrWSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
    else
        return GetImplicitGemmWrWSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx, 1);
#else
    return GetImplicitGemmWrWSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
#endif
}

std::vector<miopen::solver::ConvSolution>
FindAllBwdWrW2DSolutions(const miopen::ConvolutionContext& ctx,
                         const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetBwdWrW2DSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
}

std::vector<miopen::solver::ConvSolution>
FindAllFFTSolutions(const miopen::ConvolutionContext& ctx,
                    const miopen::AnyInvokeParams& invoke_ctx)
{
    return GetFFTSolvers().SearchForAllSolutions(ctx, GetDb(ctx), invoke_ctx);
}

std::vector<std::pair<std::string, size_t>>
AllFFTForwardBackwardDataWorkspaceSize(const miopen::ConvolutionContext& ctx)
{
    return GetFFTSolvers().GetWorkspaceSize(ctx);
}

void miopen::ConvolutionContext::SetupFloats()
{
    if(IsFp32() || IsFp16() || IsBfp16())
    {
        general_compile_options += GetDataTypeKernelParams(in_data_type);
    }
    else
    {
        MIOPEN_LOG_W("Unsupported data types configuration: "
                     << miopen::GetDataTypeName(in_data_type) << "x"
                     << miopen::GetDataTypeName(weights_data_type) << "x"
                     << miopen::GetDataTypeName(out_data_type));
    }
}

void mlo_construct_activ_lrn_pooling_common::setupFloats()
{
    if(_search_params.in_data_type == miopenFloat && _search_params.out_data_type == miopenFloat)
    {
        _search_params.general_compile_options += " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";
    }
    else if(_search_params.in_data_type == miopenHalf && _search_params.out_data_type == miopenHalf)
    {
        _search_params.general_compile_options += " -DMIOPEN_USE_FP32=0 -DMIOPEN_USE_FP16=1";
    }
    else
    {
        MIOPEN_LOG_W("Unsupported data types configuration: "
                     << miopen::GetDataTypeName(_search_params.in_data_type) << "x"
                     << miopen::GetDataTypeName(_search_params.out_data_type));
    }
}
