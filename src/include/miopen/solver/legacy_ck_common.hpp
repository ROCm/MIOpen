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
#include <sstream>

#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/stringutils.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_LEGACY_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_LEGACY_CK_USE_AMD_BUFFER_ADDRESSING)

namespace miopen {
namespace solver {
namespace legacy_ck {

/// \todo Check which devices are currently supported
static inline bool is_ck_supported_hardware(const Handle& handle)
{
    return (StartsWith(handle.GetDeviceName(), "gfx803") && handle.GetMaxComputeUnits() == 64) ||
           StartsWith(handle.GetDeviceName(), "gfx900") ||
           StartsWith(handle.GetDeviceName(), "gfx906") ||
           StartsWith(handle.GetDeviceName(), "gfx908") ||
           StartsWith(handle.GetDeviceName(), "gfx90a") ||
           StartsWith(handle.GetDeviceName(), "gfx942") ||
           StartsWith(handle.GetDeviceName(), "gfx950") ||
           StartsWith(handle.GetDeviceName(), "gfx1030") ||
           StartsWith(handle.GetDeviceName(), "gfx1031") ||
           StartsWith(handle.GetDeviceName(), "gfx1100") ||
           StartsWith(handle.GetDeviceName(), "gfx1101") ||
           StartsWith(handle.GetDeviceName(), "gfx1102") ||
           StartsWith(handle.GetDeviceName(), "gfx1151") ||
           StartsWith(handle.GetDeviceName(), "gfx1200") ||
           StartsWith(handle.GetDeviceName(), "gfx1201");
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

    // device_name: "gfx90a" -> macro: "CK_AMD_GPU_GFX90A"
    if(StartsWith(device_name, "gfx"))
    {
        std::string gfxid = device_name.substr(3);
        std::transform(gfxid.begin(), gfxid.end(), gfxid.begin(), ::toupper);
        compiler_flag << " -DCK_AMD_GPU_GFX" << gfxid;
    }

    // buffer atomic-fadd
    compiler_flag << " -DCK_USE_AMD_BUFFER_ATOMIC_FADD="
                  << (is_support_amd_buffer_atomic_fadd(device_name) ? '1' : '0');

    // sync LDS
    compiler_flag << " -DCK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM="
                  << (env::disabled(MIOPEN_DEBUG_LEGACY_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM) ? '0'
                                                                                             : '1');

    // buffer addressing
    compiler_flag << " -DCK_USE_AMD_BUFFER_ADDRESSING="
                  << (env::disabled(MIOPEN_DEBUG_LEGACY_CK_USE_AMD_BUFFER_ADDRESSING) ? '0' : '1');

    return compiler_flag.str();
}

} // namespace legacy_ck
} // namespace solver
} // namespace miopen
