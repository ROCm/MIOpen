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

#if MIOPEN_ENABLE_SQLITE
miopen::PerformanceDb mlo_construct_base::GetDb() const
{
    auto& h = _search_params.GetStream();
    return {
        db_path(), _search_params.GetUserPerfDbPath(), h.GetDeviceName(), h.GetMaxComputeUnits()};
}
miopen::PerformanceDb miopen::GetDb(const miopen::ConvolutionContext& ctx)
{
    auto& h = ctx.GetStream();
    return {
        ctx.GetPerfDbPath(), ctx.GetUserPerfDbPath(), h.GetDeviceName(), h.GetMaxComputeUnits()};
}
#else
miopen::PerformanceDb mlo_construct_base::GetDb() const
{
    return {db_path(), _search_params.GetUserPerfDbPath()};
}

miopen::PerformanceDb miopen::GetDb(const ConvolutionContext& ctx)
{
    return {ctx.GetPerfDbPath(), ctx.GetUserPerfDbPath()};
}
#endif
miopen::solver::ConvSolution
mlo_construct_direct2D_fusion::FindSolution(const std::vector<miopen::solver::AnySolver>& solvers)
{
    miopen::solver::ConvSolution solution{miopenStatusUnknownError};
    std::string solver_id;
    auto db = this->GetDb();
    for(auto& solver : solvers)
    {
        solution = solver.FindSolution(_search_params, db);
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
                                           miopen::solver::ConvOclDirectFwd3x3,
                                           miopen::solver::ConvOclDirectFwd1x1,
                                           miopen::solver::ConvOclDirectFwd>{};
}

static auto GetImplicitGemmSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvHipImplicitGemmV4R4Xdlops_1x1,
                                           miopen::solver::ConvHipImplicitGemmV4R4GenFwdXdlops,
                                           miopen::solver::ConvHipImplicitGemmV4R4FwdXdlops,
                                           miopen::solver::ConvHipImplicitGemmBwdDataV1R1Xdlops,
                                           miopen::solver::ConvHipImplicitGemmV4_1x1,
                                           miopen::solver::ConvHipImplicitGemmV4Fwd,
                                           miopen::solver::ConvHipImplicitGemmV4R1Fwd,
                                           miopen::solver::ConvHipImplicitGemmV4R4Fwd,
                                           miopen::solver::ConvHipImplicitGemmBwdDataV1R1,
                                           miopen::solver::ConvHipImplicitGemmBwdDataV4R1>{};
}

static auto GetWindogradSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvBinWinograd3x3U,
                                           miopen::solver::ConvBinWinogradRxSf3x2,
                                           miopen::solver::ConvBinWinogradRxSf2x3,
                                           miopen::solver::ConvBinWinogradRxS>{};
}

static auto GetImplicitGemmWrWSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvHipImplicitGemmV4R4WrWXdlops,
                                           miopen::solver::ConvHipImplicitGemmV4R4GenWrWXdlops,
                                           miopen::solver::ConvHipImplicitGemmV4WrW,
                                           miopen::solver::ConvHipImplicitGemmV4R1WrW>{};
}

static auto GetWindogradWrWSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvBinWinogradRxS,
                                           miopen::solver::ConvBinWinogradRxSf2x3,
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
                                           miopen::solver::ConvOclBwdWrW1x1>{};
}

#if MIOPEN_USE_SCGEMM
static auto GetFwdSCGemmSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvSCGemmFGemm>{};
}
#endif

std::vector<miopen::solver::ConvSolution>
FindAllDirectSolutions(const miopen::ConvolutionContext& ctx)
{
    return GetDirectSolvers().SearchForAllSolutions(ctx, GetDb(ctx));
}

std::vector<std::pair<std::string, size_t>>
AllDirectForwardBackwardDataWorkspaceSize(const miopen::ConvolutionContext& ctx)
{
    return GetDirectSolvers().GetWorkspaceSize(ctx);
}

std::vector<miopen::solver::ConvSolution>
FindAllImplicitGemmSolutions(const miopen::ConvolutionContext& ctx)
{
    return GetImplicitGemmSolvers().SearchForAllSolutions(ctx, GetDb(ctx));
}

std::vector<miopen::solver::ConvSolution>
FindAllWinogradSolutions(const miopen::ConvolutionContext& ctx)
{
    return GetWindogradSolvers().SearchForAllSolutions(ctx, GetDb(ctx));
}

std::vector<miopen::solver::ConvSolution>
FindWinogradWrWAllSolutions(const miopen::ConvolutionContext& ctx)
{
    return GetWindogradWrWSolvers().SearchForAllSolutions(ctx, GetDb(ctx));
}

std::vector<std::pair<std::string, size_t>>
AllDirectBwdWrW2DWorkspaceSize(const miopen::ConvolutionContext& ctx)
{
    return GetBwdWrW2DSolvers().GetWorkspaceSize(ctx);
}

std::vector<miopen::solver::ConvSolution>
FindImplicitGemmWrWAllSolutions(const miopen::ConvolutionContext& ctx)
{
    return GetImplicitGemmWrWSolvers().SearchForAllSolutions(ctx, GetDb(ctx));
}

std::vector<miopen::solver::ConvSolution>
FindAllBwdWrW2DSolutions(const miopen::ConvolutionContext& ctx)
{
    return GetBwdWrW2DSolvers().SearchForAllSolutions(ctx, GetDb(ctx));
}

std::vector<miopen::solver::ConvSolution>
FindAllFwdSCGemmSolutions(const miopen::ConvolutionContext& ctx)
{
#if MIOPEN_USE_SCGEMM
    return GetFwdSCGemmSolvers().SearchForAllSolutions(ctx, GetDb(ctx));
#else
    (void)ctx;
    return {};
#endif
}

void miopen::ConvolutionContext::SetupFloats()
{
    if(IsFp32() || IsFp16() || IsBfp16())
    {
        general_compile_options += GetDataTypeKernelParams(in_data_type);
    }
    else
    {
        MIOPEN_LOG_W(
            "Unsupported data types configuration: " << miopen::GetDataTypeName(in_data_type) << "x"
                                                     << miopen::GetDataTypeName(weights_data_type)
                                                     << "x"
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
                     << miopen::GetDataTypeName(_search_params.in_data_type)
                     << "x"
                     << miopen::GetDataTypeName(_search_params.out_data_type));
    }
}
