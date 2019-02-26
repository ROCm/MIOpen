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

#include <cmath>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <unordered_map>

#include <miopen/solver.hpp>
#include <miopen/db.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/mlo_utils.hpp>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_KERNELS)

/************************************************************************************************************************
 **
 **			CONSTRUCT CONVOLUTIONAL LAYER
 **
 ************************************************************************************************************************/

miopen::MultiFileDb mlo_construct_direct2D::GetDb() const
{
    return {db_path(), _search_params.GetUserPerfDbPath()};
}

miopen::solver::ConvSolution
mlo_construct_direct2D_fusion::FindSolution(std::vector<miopen::solver::AnySolver> solvers)
{
    miopen::solver::ConvSolution solution{miopenStatusUnknownError};
    std::string solver_id;
    auto db = this->GetDb();
    for(auto& solver : solvers)
    {
        solution = solver.FindSolution(_search_params, db);
        if(solution.Succeeded() && solver.IsApplicable(_search_params) &&
           solver.IsFast(_search_params))
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

std::vector<miopen::solver::ConvSolution> mlo_construct_direct2D::FindAllSolutions()
{
    // clang-format off
    return miopen::solver::SearchForAllSolutions<
        miopen::solver::ConvAsm3x3U,
        miopen::solver::ConvAsm1x1U,
        miopen::solver::ConvAsm5x10u2v2f1,
        miopen::solver::ConvAsm7x7c3h224w224k64u2v2p3q3f1,
        miopen::solver::ConvAsm5x10u2v2b1,
        miopen::solver::ConvOclDirectFwd11x11,
        miopen::solver::ConvOclDirectFwdGen,
        miopen::solver::ConvOclDirectFwd3x3,
        miopen::solver::ConvOclDirectFwd1x1,
        miopen::solver::ConvOclDirectFwd
    >(_search_params, this->GetDb());
    // clang-format on
}

miopen::solver::ConvSolution mlo_construct_winograd::FindSolution()
{
    // clang-format off
    return miopen::solver::SearchForSolution<
        miopen::solver::ConvBinWinograd3x3U,
        miopen::solver::ConvBinWinogradRxS
    >(_search_params, this->GetDb());
    // clang-format on
}

miopen::solver::ConvSolution mlo_construct_winograd_wrw::FindSolution()
{
    // clang-format off
    return miopen::solver::SearchForSolution<
        miopen::solver::ConvBinWinogradRxS
    >(_search_params, this->GetDb());
    // clang-format on
}

std::vector<miopen::solver::ConvSolution> mlo_construct_BwdWrW2D::FindAllSolutions()
{
    // clang-format off
    return miopen::solver::SearchForAllSolutions<
        miopen::solver::ConvAsmBwdWrW1x1,
        miopen::solver::ConvAsmBwdWrW3x3,
        miopen::solver::ConvOclBwdWrW2<1>,
        miopen::solver::ConvOclBwdWrW2<2>,
        miopen::solver::ConvOclBwdWrW2<4>,
        miopen::solver::ConvOclBwdWrW2<8>,
        miopen::solver::ConvOclBwdWrW2<16>,
        miopen::solver::ConvOclBwdWrW2NonTunable,
        miopen::solver::ConvOclBwdWrW53,
        miopen::solver::ConvOclBwdWrW1x1
    >(_search_params, this->GetDb());
    // clang-format on
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

static std::ostream& operator<<(std::ostream& os, const rocm_meta_version& rmv)
{
    switch(rmv)
    {
    case rocm_meta_version::Unknown: return os << "Unknown";
    case rocm_meta_version::V1: return os << "V1";
    case rocm_meta_version::V2: return os << "V2";
    case rocm_meta_version::V3: return os << "V3";
    case rocm_meta_version::AMDHSA_1_0: return os << "AMDHSA_1_0";
    }
    return os << "<Error>";
}

static rocm_meta_version DetectAmdRocmMetadataVersion(const miopen::ConvolutionContext& context)
{
#if MIOPEN_BACKEND_OPENCL
    const auto dev                     = miopen::GetDevice(context.GetStream().GetStream());
    const auto platform                = miopen::GetDeviceInfo<CL_DEVICE_PLATFORM>(dev);
    const std::string platform_version = miopen::GetPlatformInfo<CL_PLATFORM_VERSION>(
        platform); // e.g. "OpenCL 2.0 AMD-APP.internal (2334.0)"
    size_t num_begin      = platform_version.find('(');
    rocm_meta_version rmv = rocm_meta_version::Unknown;
    if(num_begin != std::string::npos)
    {
        int num = std::stoi(platform_version.substr(num_begin + 1));
        if(num < 2338) // Switched to V2 somewhere within [2337,2338]
            rmv = rocm_meta_version::V1;
        else if(num < 2389) // Switched to V3 somewhere within [2388,2389]
            rmv = rocm_meta_version::V2;
        else if(num < 2535) // Switched to newer version at 2535 for sure.
            rmv = rocm_meta_version::V3;
        else
            rmv = rocm_meta_version::AMDHSA_1_0;
    }
#else
    /// \todo Rework this using clang-ocl.
    (void)context;
    rocm_meta_version rmv = rocm_meta_version::Default;
    // Assembler is always available for HIP backend.
    // ROCm 1.7, which uses AMDHSA_1_0 metadata, does not have bug 34765 in
    // the assembler. Previous ROCm versions have this bug.
    if(!GcnAssemblerHasBug34765())
    {
        rmv = rocm_meta_version::AMDHSA_1_0;
    }
#endif // MIOPEN_BACKEND_OPENCL
    MIOPEN_LOG_I(rmv);
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
        static const rocm_meta_version ret_rmv = DetectAmdRocmMetadataVersion(context);
        context.rmv                            = ret_rmv;
    }
    return ret_bool;
}

void mlo_construct_direct2D::setupFloats()
{
    if(_search_params.IsFp32())
    {
        _search_params.general_compile_options += " -DMIOPEN_USE_FP32=1 -DMIOPEN_USE_FP16=0";
    }
    else if(_search_params.IsFp16())
    {
        _search_params.general_compile_options += " -DMIOPEN_USE_FP32=0 -DMIOPEN_USE_FP16=1";
    }
    else
    {
        MIOPEN_LOG_W("Unsupported data types configuration: "
                     << miopen::GetDataTypeName(_search_params.in_data_type)
                     << "x"
                     << miopen::GetDataTypeName(_search_params.weights_data_type)
                     << "x"
                     << miopen::GetDataTypeName(_search_params.out_data_type));
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

void mlo_construct_direct2D::setupRocm()
{
    // Detect assembly kernels
    _search_params.use_binaries    = false;
    _search_params.use_asm_kernels = false;
    _search_params.rmv             = rocm_meta_version::Default;
    if(mloIsAmdRocmOpencl(_search_params))
    {
        _search_params.use_asm_kernels =
            !miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_KERNELS{}) && ValidateGcnAssembler();
#ifndef HIP_OC_FINALIZER
        _search_params.use_binaries =
            !miopen::IsDisabled(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES{});
#endif
    }
}

bool mlo_construct_direct2D::mloIsFastBinaryWinograd3x3U() const
{
    return (_search_params.n_outputs >= 16 && _search_params.n_outputs % 2 == 0);
}
