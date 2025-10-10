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

#include <miopen/env.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/stringutils.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_STATIC_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM)

#define WORKAROUND_SWDEV_229277_227616_229195 1
// workaround for unnecessary VGPA <--> AGRP data movement when using mfma LLVM intrinsic
#define WORKAROUND_SWDEV_229564 1
// workaround for buffer load/store fp16/bfp16 intrinsic bug
#define WORKAROUND_SWDEV_231101 1
// due to compiler bug, iGEMM xdlops kernels fail verification in some cases, if using "-O3" flag,
// (but will pass verification with "-O1" flag)
#define WORKAROUND_SWDEV_251757 1
// although gfx1030 supports buffer instructions,but it not work properly when we use the
// corresponding llvm intrinsic functions
// so we disable using those llvm intrinsic functions on gfx1030
#define WORKAROUND_MIOPEN_ISSUE_557 1
#define WORKAROUND_SWDEV_413051 1

namespace miopen {
namespace solver {
namespace static_ck {

/// \todo Check which devices are currently supported
static inline bool IsComposableKernelSupportedHardware(const ExecutionContext& c)
{
    return (c.GetStream().GetDeviceName() == "gfx803" &&
            c.GetStream().GetMaxComputeUnits() == 64) ||
           c.GetStream().GetDeviceName() == "gfx900" || c.GetStream().GetDeviceName() == "gfx906" ||
           c.GetStream().GetDeviceName() == "gfx908" || c.GetStream().GetDeviceName() == "gfx90a" ||
           c.GetStream().GetDeviceName() == "gfx942" ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx95") ||
           StartsWith(c.GetStream().GetDeviceName(), "gfx103");
}

static inline bool GfxHasMissingBf16Intrinsics(const std::string& device_name)
{
    return device_name == "gfx942" || StartsWith(device_name, "gfx95");
}

static inline bool GfxHasMissingFp32Intrinsics(const std::string& device_name)
{
    return device_name == "gfx942" || StartsWith(device_name, "gfx95");
}

static inline bool support_amd_buffer_atomic_fadd(const std::string& device_name)
{
    return StartsWith(device_name, "gfx908");
}

static inline bool is_use_v_fmac_f32(const ExecutionContext& ctx)
{
    const auto device_name = ctx.GetStream().GetDeviceName();
    return StartsWith(device_name, "gfx103");
}

static inline bool is_use_amd_buffer_load_store(const ExecutionContext& ctx)
{
#if WORKAROUND_MIOPEN_ISSUE_557
    const auto device_name = ctx.GetStream().GetDeviceName();
    return !StartsWith(device_name, "gfx103");
#else
    return true;
#endif
}

template <typename T>
int amd_buffer_load_max_length()
{
    if(std::is_same<float, T>() || WORKAROUND_SWDEV_413051)
    {
        return 4;
    }
    else if(std::is_same<half_float::half, T>())
    {
        return 8;
    }
    else
    {
        MIOPEN_LOG_I("not implemented");
        return 1;
    }
}

template <typename T>
int amd_buffer_store_max_length()
{
    if(std::is_same<float, T>())
    {
        return 4;
    }
    else if(std::is_same<half_float::half, T>())
    {
        return 8;
    }
    else
    {
        MIOPEN_LOG_I("not implemented");
        return 1;
    }
}

template <typename T>
int amd_lds_read_max_length()
{
    if(std::is_same<float, T>())
    {
        return 4;
    }
    else if(std::is_same<half_float::half, T>())
    {
        return 8;
    }
    else
    {
        MIOPEN_LOG_I("not implemented");
        return 1;
    }
}

template <typename T>
int amd_lds_write_max_length()
{
    if(std::is_same<float, T>())
    {
        return 4;
    }
    else if(std::is_same<half_float::half, T>())
    {
        return 8;
    }
    else
    {
        MIOPEN_LOG_I("not implemented");
        return 1;
    }
}

constexpr std::size_t get_lds_max_number_of_byte() { return 65536; }

static inline auto get_static_ck_common_compiler_flag(const ExecutionContext& ctx)
{
    auto compiler_flag     = std::string(" --std=c++14");
    auto buffer_atomic_add = support_amd_buffer_atomic_fadd(ctx.GetStream().GetDeviceName());

    // atomic-fadd
    compiler_flag +=
        std::string(" -DCK_USE_AMD_BUFFER_ATOMIC_FADD=") + (buffer_atomic_add ? '1' : '0');

    if(buffer_atomic_add)
        compiler_flag += std::string(" -DCK_AMD_BUFFER_ATOMIC_FADD_RETURNS_FLOAT=1");

    // LDS sync
    compiler_flag +=
        std::string(" -DCK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM=") +
        (env::disabled(MIOPEN_DEBUG_STATIC_CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM) ? '0' : '1');

    // workaround
    compiler_flag +=
        std::string(" -DCK_WORKAROUND_SWDEV_229564=") + std::to_string(WORKAROUND_SWDEV_229564) +
        std::string(" -DCK_WORKAROUND_SWDEV_231101=") + std::to_string(WORKAROUND_SWDEV_231101);

    // enable or disable buffer load/store
    compiler_flag += std::string(" -DCK_USE_AMD_BUFFER_ADDRESSING=") +
                     (is_use_amd_buffer_load_store(ctx) ? '1' : '0');

    // use v_fmac_f32 or not
    compiler_flag +=
        std::string(" -DCK_USE_AMD_V_FMAC_F32=") + (is_use_v_fmac_f32(ctx) ? '1' : '0');

    return compiler_flag;
}

} // namespace static_ck
} // namespace solver
} // namespace miopen
