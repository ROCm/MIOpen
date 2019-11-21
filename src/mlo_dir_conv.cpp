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

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_KERNELS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_OPENCL_CONVOLUTIONS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_NEWER)

// Only select first applicable implicitgemm kernel due to slow compilation time
// (issue SWDEV-201055) and tuning
/// \todo enable multiple or all applicable solver search after fixing slow compilation
#define IMPLICIT_GEMM_FIND_FIRST_SOLUTION 1
#if MIOPEN_ENABLE_SQLITE
miopen::PerfDb mlo_construct_base::GetDb() const
{
    auto& h = _search_params.GetStream();
    return {
        db_path(), _search_params.GetUserPerfDbPath(), h.GetDeviceName(), h.GetMaxComputeUnits()};
}
miopen::PerfDb miopen::GetDb(const miopen::ConvolutionContext& ctx)
{
    auto& h = ctx.GetStream();
    return {
        ctx.GetPerfDbPath(), ctx.GetUserPerfDbPath(), h.GetDeviceName(), h.GetMaxComputeUnits()};
}
#else
miopen::PerfDb mlo_construct_base::GetDb() const
{
    return {db_path(), _search_params.GetUserPerfDbPath()};
}

miopen::PerfDb miopen::GetDb(const ConvolutionContext& ctx)
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
                                           miopen::solver::ConvHipImplicitGemmV4R4FwdXdlops,
                                           miopen::solver::ConvHipImplicitGemmV4_1x1,
                                           miopen::solver::ConvHipImplicitGemmV4Fwd,
                                           miopen::solver::ConvHipImplicitGemmV4R1Fwd>{};
}

static auto GetWindogradSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvBinWinograd3x3U,
                                           miopen::solver::ConvBinWinogradRxSf3x2,
                                           miopen::solver::ConvBinWinogradRxS>{};
}

static auto GetImplicitGemmWrWSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvHipImplicitGemmV4R4WrWXdlops,
                                           miopen::solver::ConvHipImplicitGemmV4WrW,
                                           miopen::solver::ConvHipImplicitGemmV4R1WrW>{};
}

static auto GetWindogradWrWSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::ConvBinWinogradRxS,
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
#if IMPLICIT_GEMM_FIND_FIRST_SOLUTION
    return GetImplicitGemmSolvers().SearchForAllSolutions(ctx, GetDb(ctx), 1);
#else
    return GetImplicitGemmSolvers().SearchForAllSolutions(ctx, GetDb(ctx));
#endif
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
#if IMPLICIT_GEMM_FIND_FIRST_SOLUTION
    return GetImplicitGemmWrWSolvers().SearchForAllSolutions(ctx, GetDb(ctx), 1);
#else
    return GetImplicitGemmWrWSolvers().SearchForAllSolutions(ctx, GetDb(ctx));
#endif
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

#if MIOPEN_BACKEND_OPENCL
static bool IsTokenWithin(const std::string& s, const char* delimiters, const std::string& find_tok)
{
    assert(delimiters);
    std::size_t cursor = 0;
    do
    {
        const std::size_t tok_begin = s.find_first_not_of(delimiters, cursor);
        if(tok_begin == std::string::npos)
        {
            break;
        }
        cursor            = s.find_first_of(delimiters, tok_begin);
        std::string token = (cursor == std::string::npos) ? s.substr(tok_begin)
                                                          : s.substr(tok_begin, cursor - tok_begin);
        if(token == find_tok)
        {
            return true;
        }
    } while(cursor != std::string::npos);
    return false;
}

static bool IsAmdRocmOpencl(const miopen::ConvolutionContext& context)
{
    const auto dev             = miopen::GetDevice(context.GetStream().GetStream());
    const auto platform        = miopen::GetDeviceInfo<CL_DEVICE_PLATFORM>(dev);
    const auto platform_vendor = miopen::GetPlatformInfo<CL_PLATFORM_VENDOR>(platform);
    if(platform_vendor != "Advanced Micro Devices, Inc.")
    {
        return false;
    }
    const auto device_vendor_id = miopen::GetDeviceInfo<CL_DEVICE_VENDOR_ID>(dev);
    if(device_vendor_id != 0x1002) // AMD
    {
        return false;
    }
    const auto driver_version = miopen::GetDeviceInfo<CL_DRIVER_VERSION>(dev);
    const char* delimiters    = " (),*";                    // Specific for ROCm OCL driver version.
    return IsTokenWithin(driver_version, delimiters, "LC"); // Lightning Compiler.
}
#endif // MIOPEN_BACKEND_OPENCL

/// This is intended to use only in Asm Solvers which support both CO v2 and CO v3.
/// It says which code object format shall be selected during the build process.
///
/// If ROCm supports only v2 or v3, the answer is trivial. When Solver supports
/// single CO version, the logic is trivial as well.
///
/// However, when both ROCm and Solver are able to support both code object formats,
/// these is no objective criterion for making a decision. The following behavior
/// is implemented:
/// * By default, the older format is used (CO v2).
/// * If MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_NEWER is set to 1, then
///   the behavior is reversed and CO v3 is selected.
///
/// FIXME move this out of the rocm_meta_version class.
bool rocm_meta_version::UseV3() const
{
    if(miopen::IsEnabled(MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_NEWER{}))
        return val == AMDHSA_COv3 || val == AMDHSA_COv2_COv3;
    else
        return val == AMDHSA_COv3;
}

static std::ostream& operator<<(std::ostream& os, const rocm_meta_version& rmv)
{
    switch(rmv.getValue())
    {
    case rocm_meta_version::Unknown: return os << "Unknown";
    case rocm_meta_version::AMDHSA_COv2: return os << "AMDHSA_COv2";
    case rocm_meta_version::AMDHSA_COv2_COv3: return os << "AMDHSA_COv2_COv3";
    case rocm_meta_version::AMDHSA_COv3: return os << "AMDHSA_COv3";
    default: break;
    }
    return os << "<Error>";
}

static rocm_meta_version AmdRocmMetadataVersionGetEnv()
{
    const rocm_meta_version val(
        static_cast<int>(miopen::Value(MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE{})));
    if(!val.IsValid())
    {
        MIOPEN_LOG_W("Incorrect MIOPEN_DEBUG_AMD_ROCM_ENFORCE_MDVERSION = " << val.getValue()
                                                                            << ", using default.");
        return rocm_meta_version::Unknown;
    }
    return val;
}

static rocm_meta_version AmdRocmMetadataVersionDetect(const miopen::ConvolutionContext& context)
{
    rocm_meta_version rmv = AmdRocmMetadataVersionGetEnv();
    if(rmv.IsUnknown())
    {
#if MIOPEN_BACKEND_OPENCL
        const auto dev                     = miopen::GetDevice(context.GetStream().GetStream());
        const auto platform                = miopen::GetDeviceInfo<CL_DEVICE_PLATFORM>(dev);
        const std::string platform_version = miopen::GetPlatformInfo<CL_PLATFORM_VERSION>(
            platform); // e.g. "OpenCL 2.0 AMD-APP.internal (2334.0)"
        size_t num_begin = platform_version.find('(');
        if(num_begin != std::string::npos)
        {
            // int num = std::stoi(platform_version.substr(num_begin + 1));
            rmv = rocm_meta_version::AMDHSA_COv2;
        }
        else
        {
            rmv = rocm_meta_version::Default;
        }
#else
        (void)context;
        rmv = rocm_meta_version::Default;
        /// This is only to print information onto console.
        /// \todo Consider removing this call in installable builds.
        (void)miopen::HipGetHccVersion();
#endif // MIOPEN_BACKEND_OPENCL
    }
    MIOPEN_LOG_NQI(
        "ROCm MD version "
        << rmv
        << ", MIOpen version " MIOPEN_STRINGIZE(MIOPEN_VERSION_MAJOR) "." MIOPEN_STRINGIZE(
               MIOPEN_VERSION_MINOR) "." MIOPEN_STRINGIZE(MIOPEN_VERSION_PATCH) "." MIOPEN_STRINGIZE(MIOPEN_VERSION_TWEAK));
    return rmv;
}

static bool mloIsAmdRocmOpencl(miopen::ConvolutionContext& context)
{
    static const bool ret_bool =
#if MIOPEN_BACKEND_OPENCL
        IsAmdRocmOpencl(context);
#else
        true;
#endif // MIOPEN_BACKEND_OPENCL
    if(ret_bool)
    {
        static const rocm_meta_version ret_rmv = AmdRocmMetadataVersionDetect(context);
        context.rmv                            = ret_rmv;
    }
    return ret_bool;
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

void miopen::ConvolutionContext::DetectRocm()
{
    // Detect assembly kernels
    use_binaries            = false;
    use_asm_kernels         = false;
    use_opencl_convolutions = !miopen::IsDisabled(MIOPEN_DEBUG_OPENCL_CONVOLUTIONS{});
    rmv                     = rocm_meta_version::Default;
    if(mloIsAmdRocmOpencl(*this))
    {
        use_asm_kernels =
            !miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_KERNELS{}) && ValidateGcnAssembler();
#ifndef HIP_OC_FINALIZER
        use_binaries = !miopen::IsDisabled(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES{});
#endif
    }

    if(StartsWith(GetStream().GetDeviceName(), "gfx8"))
    {
        use_asm_kernels = false;
        use_binaries    = false;
    }
}
