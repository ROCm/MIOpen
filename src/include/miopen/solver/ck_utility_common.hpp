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
#include <miopen/rocm_features.hpp>
#include <miopen/solver/convolution_context_interpreter.hpp>
#include <algorithm>
#include <sstream>

#include "../composable_kernel/composable_kernel/include/utility/data_type_enum.hpp"
#include "../composable_kernel/host/solver/include/convolution_problem_descriptor.hpp"
#include "../composable_kernel/host/solver/include/solver_common.hpp"

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CK_USE_AMD_BUFFER_ADDRESSING)

namespace miopen {
namespace solver {
namespace ck_utility {

static inline bool is_ck_supported_hardware(const Handle& handle)
{
    return (StartsWith(handle.GetDeviceName(), "gfx803") && handle.GetMaxComputeUnits() == 64) ||
           StartsWith(handle.GetDeviceName(), "gfx900") ||
           StartsWith(handle.GetDeviceName(), "gfx906") ||
           StartsWith(handle.GetDeviceName(), "gfx908") ||
           StartsWith(handle.GetDeviceName(), "gfx90a") ||
           StartsWith(handle.GetDeviceName(), "gfx1031") ||
           StartsWith(handle.GetDeviceName(), "gfx1030");
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
    else if(StartsWith(device_name, "gfx1030"))
        compiler_flag << " -DCK_AMD_GPU_GFX1030";
    else if(StartsWith(device_name, "gfx1031"))
        compiler_flag << " -DCK_AMD_GPU_GFX1031";

    // buffer atomic-fadd
    compiler_flag << " -DCK_USE_AMD_BUFFER_ATOMIC_FADD="
                  << (is_support_amd_buffer_atomic_fadd(device_name) ? '1' : '0');

    // sync LDS
    compiler_flag << " -DCK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM="
                  << (miopen::IsDisabled(MIOPEN_DEBUG_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM{}) ? '0'
                                                                                             : '1');

    // buffer addressing
    compiler_flag << " -DCK_USE_AMD_BUFFER_ADDRESSING="
                  << (miopen::IsDisabled(MIOPEN_DEBUG_CK_USE_AMD_BUFFER_ADDRESSING{}) ? '0' : '1');

    return compiler_flag.str();
}

static inline auto get_ck_convolution_problem_descriptor(const ConvolutionContext& ctx)
{
    ck::DataTypeEnum_t ck_datatype;

    if(ctx.IsFp32())
        ck_datatype = ck::DataTypeEnum_t::Float;
    else if(ctx.IsFp16())
        ck_datatype = ck::DataTypeEnum_t::Half;
    else if(ctx.IsBfp16())
        ck_datatype = ck::DataTypeEnum_t::BFloat16;
    else
        ck_datatype = ck::DataTypeEnum_t::Unknown;

    return ck::driver::ConvolutionProblemDescriptor{
        miopen::solver::ConvolutionContextInterpreter::GetBatchN(ctx),
        ConvolutionContextInterpreter::GetOutputChannelK(ctx),
        ConvolutionContextInterpreter::GetInputChannelC(ctx),
        ConvolutionContextInterpreter::GetFilterHeightY(ctx),
        ConvolutionContextInterpreter::GetFilterWidthX(ctx),
        ConvolutionContextInterpreter::GetInputHeightHi(ctx),
        ConvolutionContextInterpreter::GetInputWidthWi(ctx),
        ConvolutionContextInterpreter::GetOutputHeightHo(ctx),
        ConvolutionContextInterpreter::GetOutputWidthWo(ctx),
        ConvolutionContextInterpreter::GetAdjustedConvolutionStrideH(ctx),
        ConvolutionContextInterpreter::GetAdjustedConvolutionStrideW(ctx),
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationH(ctx),
        ConvolutionContextInterpreter::GetAdjustedConvolutionDilationW(ctx),
        ConvolutionContextInterpreter::GetInputLeftPadH(ctx),
        ConvolutionContextInterpreter::GetInputLeftPadW(ctx),
        ConvolutionContextInterpreter::GetAdjustedInputRightPadH(ctx),
        ConvolutionContextInterpreter::GetAdjustedInputRightPadW(ctx),
        ck_datatype,
        ck_datatype,
        ck_datatype};
}

} // namespace ck_utility
} // namespace solver
} // namespace miopen

#endif
