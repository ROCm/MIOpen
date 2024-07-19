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

#ifndef GUARD_CK_UTILITY_COMMON_HPP_
#define GUARD_CK_UTILITY_COMMON_HPP_

#include <miopen/env.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <algorithm>
#include <sstream>

#include "../composable_kernel/composable_kernel/include/utility/data_type_enum.hpp"
#include "../composable_kernel/host/solver/include/convolution_problem_descriptor.hpp"
#include "../composable_kernel/host/solver/include/solver_common.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CK_USE_AMD_BUFFER_ADDRESSING)

namespace miopen {
namespace solver {
namespace ck_utility {

// Disclaimer: Currently CK is only supported in MI100, MI200 and MI300.
//             Please use is_ck_whitelist instead of this function.
static inline bool is_ck_supported_hardware(const Handle& handle)
{
    return (StartsWith(handle.GetDeviceName(), "gfx803") && handle.GetMaxComputeUnits() == 64) ||
           StartsWith(handle.GetDeviceName(), "gfx900") ||
           StartsWith(handle.GetDeviceName(), "gfx906") ||
           StartsWith(handle.GetDeviceName(), "gfx908") ||
           StartsWith(handle.GetDeviceName(), "gfx90a") ||
           StartsWith(handle.GetDeviceName(), "gfx940") ||
           StartsWith(handle.GetDeviceName(), "gfx941") ||
           StartsWith(handle.GetDeviceName(), "gfx942") ||
           StartsWith(handle.GetDeviceName(), "gfx1030") ||
           StartsWith(handle.GetDeviceName(), "gfx1031") ||
           StartsWith(handle.GetDeviceName(), "gfx1100") ||
           StartsWith(handle.GetDeviceName(), "gfx1101") ||
           StartsWith(handle.GetDeviceName(), "gfx1102") ||
           StartsWith(handle.GetDeviceName(), "gfx1200") ||
           StartsWith(handle.GetDeviceName(), "gfx1201");
}

// MI100 : gfx908
// MI200 : gfx90a
// MI300 : gfx940, gfx941, gfx942
static inline bool is_ck_whitelist(const std::string& device_name)
{
    return (StartsWith(device_name, "gfx908") || StartsWith(device_name, "gfx90a") ||
            StartsWith(device_name, "gfx940") || StartsWith(device_name, "gfx941") ||
            StartsWith(device_name, "gfx942"));
}

static inline bool is_ck_whitelist(const Handle& handle)
{
    return is_ck_whitelist(handle.GetDeviceName());
}

static inline bool is_support_amd_buffer_atomic_fadd(const std::string& device_name)
{
    return StartsWith(device_name, "gfx908");
}

static inline auto get_ck_common_compiler_flag(const Handle& handle)
{
    auto compiler_flag = std::stringstream();

    // C++ standard
    compiler_flag << " --std=c++17";

    // GPU target
    static const std::string device_name = handle.GetDeviceName();

    // NOLINTBEGIN(*-braces-around-statements)
    if(StartsWith(device_name, "gfx803"))
        compiler_flag << " -DCK_AMD_GPU_GFX803";
    else if(StartsWith(device_name, "gfx900"))
        compiler_flag << " -DCK_AMD_GPU_GFX900";
    else if(StartsWith(device_name, "gfx906"))
        compiler_flag << " -DCK_AMD_GPU_GFX906";
    else if(StartsWith(device_name, "gfx908"))
        compiler_flag << " -DCK_AMD_GPU_GFX908";
    else if(StartsWith(device_name, "gfx90a"))
        compiler_flag << " -DCK_AMD_GPU_GFX90A";
    else if(StartsWith(device_name, "gfx940"))
        compiler_flag << " -DCK_AMD_GPU_GFX940";
    else if(StartsWith(device_name, "gfx941"))
        compiler_flag << " -DCK_AMD_GPU_GFX941";
    else if(StartsWith(device_name, "gfx942"))
        compiler_flag << " -DCK_AMD_GPU_GFX942";
    else if(StartsWith(device_name, "gfx1030"))
        compiler_flag << " -DCK_AMD_GPU_GFX1030";
    else if(StartsWith(device_name, "gfx1031"))
        compiler_flag << " -DCK_AMD_GPU_GFX1031";
    else if(StartsWith(device_name, "gfx1100"))
        compiler_flag << " -DCK_AMD_GPU_GFX1100";
    else if(StartsWith(device_name, "gfx1101"))
        compiler_flag << " -DCK_AMD_GPU_GFX1101";
    else if(StartsWith(device_name, "gfx1102"))
        compiler_flag << " -DCK_AMD_GPU_GFX1102";
    else if(StartsWith(device_name, "gfx1200"))
        compiler_flag << " -DCK_AMD_GPU_GFX1200";
    else if(StartsWith(device_name, "gfx1201"))
        compiler_flag << " -DCK_AMD_GPU_GFX1201";
    // NOLINTEND(*-braces-around-statements)

    // buffer atomic-fadd
    compiler_flag << " -DCK_USE_AMD_BUFFER_ATOMIC_FADD="
                  << (is_support_amd_buffer_atomic_fadd(device_name) ? '1' : '0');

    // sync LDS
    compiler_flag << " -DCK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM="
                  << (env::disabled(MIOPEN_DEBUG_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM) ? '0' : '1');

    // buffer addressing
    compiler_flag << " -DCK_USE_AMD_BUFFER_ADDRESSING="
                  << (env::disabled(MIOPEN_DEBUG_CK_USE_AMD_BUFFER_ADDRESSING) ? '0' : '1');

    return compiler_flag.str();
}

static inline auto
get_ck_convolution_problem_descriptor(const miopen::conv::ProblemDescription& problem)
{
    ck::DataTypeEnum_t ck_datatype;

    // NOLINTBEGIN(*-braces-around-statements)
    if(problem.IsFp32())
        ck_datatype = ck::DataTypeEnum_t::Float;
    else if(problem.IsFp16())
        ck_datatype = ck::DataTypeEnum_t::Half;
    else if(problem.IsBfp16())
        ck_datatype = ck::DataTypeEnum_t::BFloat16;
    else
        ck_datatype = ck::DataTypeEnum_t::Unknown;
    // NOLINTEND(*-braces-around-statements)

    return ck::driver::ConvolutionProblemDescriptor{
        ProblemInterpreter::GetBatchN(problem),
        ProblemInterpreter::GetOutputChannelK(problem),
        ProblemInterpreter::GetInputChannelC(problem),
        ProblemInterpreter::GetFilterHeightY(problem),
        ProblemInterpreter::GetFilterWidthX(problem),
        ProblemInterpreter::GetInputHeightHi(problem),
        ProblemInterpreter::GetInputWidthWi(problem),
        ProblemInterpreter::GetOutputHeightHo(problem),
        ProblemInterpreter::GetOutputWidthWo(problem),
        ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
        ProblemInterpreter::GetAdjustedConvolutionStrideW(problem),
        ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
        ProblemInterpreter::GetAdjustedConvolutionDilationW(problem),
        ProblemInterpreter::GetInputLeftPadH(problem),
        ProblemInterpreter::GetInputLeftPadW(problem),
        ProblemInterpreter::GetAdjustedInputRightPadH(problem),
        ProblemInterpreter::GetAdjustedInputRightPadW(problem),
        ck_datatype,
        ck_datatype,
        ck_datatype};
}

} // namespace ck_utility
} // namespace solver
} // namespace miopen

#endif
