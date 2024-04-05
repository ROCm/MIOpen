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
#include <miopen/config.h>

#include <miopen/errors.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/hipoc_program.hpp>
#include <miopen/kernel.hpp>
#include <miopen/kernel_warnings.hpp>
#include <miopen/logger.hpp>
#include <miopen/mlir_build.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/write_file.hpp>
#include <miopen/env.hpp>
#include <miopen/comgr.hpp>
#include <boost/optional.hpp>

#include <cstring>
#include <mutex>
#include <sstream>

#if defined(__linux__)
#include <unistd.h>
#endif

/// 0 or undef or wrong - auto-detect
/// 1 - <blank> / "-Xclang -target-feature -Xclang +code-object-v3"
/// 2 - "-Xclang -target-feature -Xclang -code-object-v3" /
///     "-Xclang -target-feature -Xclang +code-object-v3"
/// 3 - "-mnocode-object-v3" / "-mcode-object-v3"
/// 4 - "-mcode-object-version=2/3/4"
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_OPTION)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_VERSION)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_USE_HIPRTC)

#if MIOPEN_USE_COMGR
#define MIOPEN_WORKAROUND_ROCM_COMPILER_SUPPORT_ISSUE_27 1
#endif

namespace miopen {

#if !MIOPEN_USE_COMGR
namespace {

int DetectCodeObjectOptionSyntax()
{
    auto syntax = miopen::Value(ENV(MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_OPTION));
    if(syntax > 4)
    {
        MIOPEN_LOG_E("Bad MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_OPTION, using default");
        syntax = 0;
    }

    if(syntax == 0)
    {
        return 4;
    }
    MIOPEN_LOG_I("MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_OPTION=" << syntax);
    return syntax;
}

int DetectCodeObjectVersion()
{
    auto co_version = miopen::Value(ENV(MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_VERSION));
    // Very basic syntax check:
    if(co_version == 1 || co_version > 4)
    {
        MIOPEN_LOG_E("Bad MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_VERSION, using default");
        co_version = 0;
    }

    if(co_version == 0)
    {
        return 4;
    }
    MIOPEN_LOG_I("MIOPEN_DEBUG_OPENCL_ENFORCE_CODE_OBJECT_VERSION=" << co_version);
    return co_version;
}

std::string GetCodeObjectVersionOptionImpl()
{
    const auto co_version = DetectCodeObjectVersion();
    const auto syntax     = DetectCodeObjectOptionSyntax();

    if(syntax == 4)
    {
        return std::string("-mcode-object-version=") + std::to_string(co_version);
    }
    else if(syntax == 3)
    {
        switch(co_version)
        {
        case 2: return {"-mnocode-object-v3"};
        default: // Fall through.
        case 3: return {"-mcode-object-v3"};
        }
    }
    else // syntax == 1 or 2
    {
        switch(co_version)
        {
        // These options are Ok for ROCm for a long time (since 2.5 or so):
        case 2: return {(syntax == 1) ? "" : "-Xclang -target-feature -Xclang -code-object-v3"};
        default: // Fall through.
        case 3: return {"-Xclang -target-feature -Xclang +code-object-v3"};
        }
    }
}

inline std::string GetCodeObjectVersionOption()
{
    static const auto option = GetCodeObjectVersionOptionImpl();
    return option;
}

} // namespace
#endif

static hipModulePtr CreateModule(const fs::path& hsaco_file)
{
    hipModule_t raw_m;
    auto status = hipModuleLoad(&raw_m, hsaco_file.string().c_str());
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed creating module from file " + hsaco_file);
    return m;
}

template <typename T> /// intended for std::string and std::vector<char>
hipModulePtr CreateModuleInMem(const T& blob)
{
    hipModule_t raw_m;
    auto status = hipModuleLoadData(&raw_m, reinterpret_cast<const void*>(blob.data()));
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed loading module");
    return m;
}

HIPOCProgramImpl::HIPOCProgramImpl(const std::string& program_name, const fs::path& filespec)
    : program(program_name), hsaco_file(filespec)
{
    module = CreateModule(hsaco_file);
}

HIPOCProgramImpl::HIPOCProgramImpl(const std::string& program_name, const std::vector<char>& blob)
    : program(program_name) ///, module(CreateModuleInMem(blob))
{
    const auto& arch = miopen::GetStringEnv(ENV(MIOPEN_DEVICE_ARCH));
    if(!arch.empty())
        return;
    module = CreateModuleInMem(blob);
}

HIPOCProgramImpl::HIPOCProgramImpl(const std::string& program_name,
                                   std::string params,
                                   const TargetProperties& target_,
                                   const std::string& kernel_src)
    : program(program_name), target(target_)
{
    BuildCodeObject(params, kernel_src);
    if(!binary.empty())
    {
        module = CreateModuleInMem(binary);
    }
    else
    {
        const auto& arch = miopen::GetStringEnv(ENV(MIOPEN_DEVICE_ARCH));
        if(arch.empty())
        {
            module = CreateModule(hsaco_file);
        }
    }
}

#if !MIOPEN_USE_COMGR
void HIPOCProgramImpl::BuildCodeObjectInFile(std::string& params,
                                             const std::string& src,
                                             const std::string& filename)
{

    dir.emplace(filename);
    hsaco_file = make_object_file_name(dir.get() / filename);

    if(miopen::EndsWith(filename, ".so"))
    {
        WriteFile(src, hsaco_file);
    }
    else if(miopen::EndsWith(filename, ".s"))
    {
        const auto assembled = AmdgcnAssemble(src, params, target);
        WriteFile(assembled, hsaco_file);
    }
    else if(miopen::EndsWith(filename, ".cpp"))
    {
        hsaco_file = HipBuild(dir, filename, src, params, target);
    }
#if MIOPEN_USE_MLIR
    else if(miopen::EndsWith(filename, ".mlir"))
    {
        std::vector<char> buffer;
        MiirGenBin(params, buffer);
        WriteFile(buffer, hsaco_file);
    }
#endif
    else
    {
        params += " " + GetCodeObjectVersionOption();
        if(miopen::IsEnabled(ENV(MIOPEN_DEBUG_OPENCL_WAVE64_NOWGP)))
            params += " -mwavefrontsize64 -mcumode";
        WriteFile(src, dir->path / filename);
        params += " -target amdgcn-amd-amdhsa -x cl -D__AMD__=1  -O3";
        params += " -cl-kernel-arg-info -cl-denorms-are-zero";
        params += " -cl-std=CL2.0 -mllvm -amdgpu-early-inline-all";
        params += " -mllvm -amdgpu-internalize-symbols ";
        params += " " + filename + " -o " + hsaco_file;
        dir->Execute(HIP_OC_COMPILER, params);
    }
    if(!fs::exists(hsaco_file))
        MIOPEN_THROW("Cant find file: " + hsaco_file);
}

#else // MIOPEN_USE_COMGR
void HIPOCProgramImpl::BuildCodeObjectInMemory(const std::string& params,
                                               const std::string& src,
                                               const std::string& filename)
{
    if(miopen::EndsWith(filename, ".so"))
    {
        std::size_t sz = src.length();
        binary.resize(sz);
        std::memcpy(&binary[0], src.c_str(), sz);
    }
    else
    {
#if MIOPEN_WORKAROUND_ROCM_COMPILER_SUPPORT_ISSUE_27
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
#endif
        if(miopen::EndsWith(filename, ".cpp"))
        {
#if MIOPEN_USE_HIPRTC
            if(!miopen::IsDisabled(ENV(MIOPEN_DEBUG_USE_HIPRTC)))
                hiprtc::BuildHip(filename, src, params, target, binary);
            else
#endif // MIOPEN_USE_HIPRTC
                comgr::BuildHip(filename, src, params, target, binary);
        }
        else if(miopen::EndsWith(filename, ".s"))
        {
            comgr::BuildAsm(filename, src, params, target, binary);
        }
#if MIOPEN_USE_MLIR
        else if(miopen::EndsWith(filename, ".mlir"))
        {
            MiirGenBin(params, binary);
        }
#endif
        else
        {
            comgr::BuildOcl(filename, src, params, target, binary);
        }
    }
    if(binary.empty())
        MIOPEN_THROW("Code object build failed. Source: " + filename);
}
#endif // MIOPEN_USE_COMGR

void HIPOCProgramImpl::BuildCodeObject(std::string params, const std::string& kernel_src)
{
    std::string filename = program;
    const auto src       = [&]() -> std::string {
        if(miopen::EndsWith(filename, ".mlir"))
            return {}; // MLIR solutions do not use source code.
        if(!kernel_src.empty())
            return kernel_src;
        return GetKernelSrc(program);
    }();

#if MIOPEN_BUILD_DEV
    if(miopen::EndsWith(filename, ".cpp"))
    {
        params += " -Werror" + HipKernelWarningsString();
    }
    else if(miopen::EndsWith(filename, ".cl"))
    {
        params += " -Werror" + OclKernelWarningsString();
    }
#else
    if(miopen::EndsWith(filename, ".cpp") || miopen::EndsWith(filename, ".cl"))
        params += " -Wno-everything";
#endif

#if MIOPEN_USE_COMGR /// \todo Refactor when functionality stabilize.
    BuildCodeObjectInMemory(params, src, filename);
#else
    BuildCodeObjectInFile(params, src, filename);
#endif
}

HIPOCProgram::HIPOCProgram() {}
HIPOCProgram::HIPOCProgram(const std::string& program_name,
                           std::string params,
                           const TargetProperties& target,
                           const std::string& kernel_src)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, params, target, kernel_src))
{
}

HIPOCProgram::HIPOCProgram(const std::string& program_name, const fs::path& hsaco)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, hsaco))
{
}

HIPOCProgram::HIPOCProgram(const std::string& program_name, const std::vector<char>& hsaco)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, hsaco))
{
}

hipModule_t HIPOCProgram::GetModule() const { return impl->module.get(); }

fs::path HIPOCProgram::GetCodeObjectPathname() const
{
    if(!impl->hsaco_file.empty())
    {
        return impl->hsaco_file;
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Empty code object path.");
    }
}

std::vector<char> HIPOCProgram::GetCodeObjectBlob() const { return impl->binary; }

void HIPOCProgram::FreeCodeObjectFileStorage()
{
    impl->dir = boost::none;
    impl->hsaco_file.clear();
}

bool HIPOCProgram::IsCodeObjectInMemory() const { return !impl->binary.empty(); };

} // namespace miopen
