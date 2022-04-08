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

#include <miopen/env.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/hip_build_utils.hpp>
#if MIOPEN_BACKEND_OPENCL
#include <miopen/ocldeviceinfo.hpp>
#endif
#include <miopen/stringutils.hpp>
#include <miopen/version.h>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_OPENCL_CONVOLUTIONS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_GCN_ASM_KERNELS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_HIP_KERNELS)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER)

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

/// This is intended to use only in Asm Solvers which support both CO v2 and CO v3.
/// It says which code object format shall be selected during the build process.
///
/// If ROCm supports only v2 or v3, the answer is trivial. When Solver supports
/// single CO version, the logic is trivial as well.
///
/// However, when both ROCm and Solver are able to support both code object formats,
/// these is no objective criterion for making a decision. The following behavior
/// is implemented:
/// * By default, the newer format is used (CO v3).
/// * If MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER is set to 1, then
///   the behavior is reversed and CO v2 is selected.
///
/// \todo Dismiss MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER (and, possibly,
/// rocm_meta_version::AMDHSA_COv2_COv3) as soon as MIOpen drops support for the
/// ROCm runtimes that can load and run both v2 and v3 Code Objects.
///
/// \todo Move this out of the rocm_meta_version class.
bool rocm_meta_version::UseV3() const
{
    if(val == AMDHSA_COv2_COv3)
        return !miopen::IsEnabled(MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER{});
    return (val == AMDHSA_COv3);
}

namespace miopen {

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

static bool CalculateIsAmdRocmOpencl(const miopen::ExecutionContext& context)
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

static rocm_meta_version AmdRocmMetadataVersionDetect(const miopen::ExecutionContext& context)
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
            const int num = std::stoi(platform_version.substr(num_begin + 1));
            if(num >= 3137) // ROCm 3.5 RC
                rmv = rocm_meta_version::AMDHSA_COv3;
            else if(num >= 3029) // ROCm 2.10 RC 1341
                rmv = rocm_meta_version::AMDHSA_COv2_COv3;
            else
                rmv = rocm_meta_version::AMDHSA_COv2;
        }
        else
        {
            rmv = rocm_meta_version::Default;
        }
#else
        (void)context;
        rmv = rocm_meta_version::Default;
#endif // MIOPEN_BACKEND_OPENCL
    }
    MIOPEN_LOG_NQI(
        "ROCm MD version "
        << rmv
        << ", HIP version " MIOPEN_STRINGIZE(HIP_PACKAGE_VERSION_MAJOR) "." MIOPEN_STRINGIZE(
               HIP_PACKAGE_VERSION_MINOR) "." MIOPEN_STRINGIZE(HIP_PACKAGE_VERSION_PATCH)
        << ", MIOpen version " MIOPEN_STRINGIZE(MIOPEN_VERSION_MAJOR) "." MIOPEN_STRINGIZE(
               MIOPEN_VERSION_MINOR) "." MIOPEN_STRINGIZE(MIOPEN_VERSION_PATCH) "." MIOPEN_STRINGIZE(MIOPEN_VERSION_TWEAK));
    return rmv;
}

static bool IsAmdRocmOpencl(miopen::ExecutionContext& context)
{
    static const bool ret_bool =
#if MIOPEN_BACKEND_OPENCL
        CalculateIsAmdRocmOpencl(context);
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

bool IsHipKernelsEnabled()
{
#if MIOPEN_USE_HIP_KERNELS
    return !miopen::IsDisabled(MIOPEN_DEBUG_HIP_KERNELS{});
#else
    return miopen::IsEnabled(MIOPEN_DEBUG_HIP_KERNELS{});
#endif
}

void miopen::ExecutionContext::DetectRocm()
{
    use_binaries            = false;
    use_asm_kernels         = false;
    use_hip_kernels         = IsHipKernelsEnabled();
    use_opencl_convolutions = !miopen::IsDisabled(MIOPEN_DEBUG_OPENCL_CONVOLUTIONS{});
    rmv                     = rocm_meta_version::Default;
    if(IsAmdRocmOpencl(*this))
    {
        use_asm_kernels =
            !miopen::IsDisabled(MIOPEN_DEBUG_GCN_ASM_KERNELS{}) && ValidateGcnAssembler();
#ifndef HIP_OC_FINALIZER
        use_binaries = !miopen::IsDisabled(MIOPEN_DEBUG_AMD_ROCM_PRECOMPILED_BINARIES{});
#endif
    }
}

} // namespace miopen
