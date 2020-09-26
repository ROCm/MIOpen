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
#include <miopen/stringutils.hpp>
#include <miopen/tmp_dir.hpp>
#include <miopen/write_file.hpp>
#include <miopen/env.hpp>
#include <miopen/comgr.hpp>
#include <boost/optional.hpp>

#include <cstring>
#include <mutex>
#include <sstream>

#include <unistd.h>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_OPENCL_ENFORCE_COV3)

#define MIOPEN_WORKAROUND_SWDEV_225285 1

#if MIOPEN_USE_COMGR
#define MIOPEN_WORKAROUND_ROCM_COMPILER_SUPPORT_ISSUE_27 1
#endif

namespace miopen {

#if !MIOPEN_USE_COMGR
namespace {

inline bool ProduceCoV3()
{
    // If env.var is set, then let's simply follow it.
    if(IsEnabled(MIOPEN_DEBUG_OPENCL_ENFORCE_COV3{}))
        return true;
    if(IsDisabled(MIOPEN_DEBUG_OPENCL_ENFORCE_COV3{}))
        return false;
    // Otherwise, let's assume that OpenCL kernels shall be compiled to
    // CO v3 format by default since ROCm 3.0. The simplest way to find out
    // this right now is checking the HIP compiler version string.
    return (HipCompilerVersion() >= external_tool_version_t{3, 0, -1});
}

/// Returns option for enabling/disabling CO v3 generation for the compiler
/// that builds OpenCL kernels, depending on compiler version etc.
inline const std::string& GetCoV3Option(const bool enable)
{
    // These options are Ok for ROCm for a long time (since 2.5 or so):
    static const std::string opt_enable{"-Xclang -target-feature -Xclang +code-object-v3"};
    static const std::string opt_disable{}; // CO v2 is compiler default.
    if(enable)
        return opt_enable;
    else
        return opt_disable;
}
} // namespace
#endif

static hipModulePtr CreateModule(const boost::filesystem::path& hsaco_file)
{
    hipModule_t raw_m;
    auto status = hipModuleLoad(&raw_m, hsaco_file.string().c_str());
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed creating module");
    return m;
}

template <typename T> /// intended for std::string and std::vector<char>
hipModulePtr CreateModuleInMem(const T& blob)
{
#if !MIOPEN_WORKAROUND_SWDEV_225285
    hipModule_t raw_m;
    auto status = hipModuleLoadData(&raw_m, reinterpret_cast<const void*>(blob.data()));
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed loading module");
    return m;
#else
    TmpDir tmp_dir("miopen");
    auto file_path = tmp_dir.path / boost::filesystem::unique_path("miopen-%%%%-%%%%-%%%%-%%%%");
    WriteFile(blob, file_path);
    return CreateModule(file_path);
#endif
}

struct HIPOCProgramImpl
{
    HIPOCProgramImpl(const std::string& program_name, const boost::filesystem::path& filespec)
        : program(program_name), hsaco_file(filespec)
    {
        module = CreateModule(hsaco_file);
    }

    HIPOCProgramImpl(const std::string& program_name, const std::string& blob)
        : program(program_name), module(CreateModuleInMem(blob))
    {
    }

    HIPOCProgramImpl(const std::string& program_name,
                     std::string params,
                     bool is_kernel_str,
                     std::string dev_name,
                     const std::string& kernel_src)
        : program(program_name), device(dev_name)
    {
        BuildCodeObject(params, is_kernel_str, kernel_src);
        if(!binary.empty())
            module = CreateModuleInMem(binary);
        else
            module = CreateModule(hsaco_file);
    }

    std::string program;
    std::string device;
    boost::filesystem::path hsaco_file;
    hipModulePtr module;
    boost::optional<TmpDir> dir;
    std::vector<char> binary;

#if !MIOPEN_USE_COMGR
    void
    BuildCodeObjectInFile(std::string& params, const std::string& src, const std::string& filename)
    {
        dir.emplace(filename);
        hsaco_file = dir->path / (filename + ".o");

        if(miopen::EndsWith(filename, ".so"))
        {
            WriteFile(src, hsaco_file);
        }
        else if(miopen::EndsWith(filename, ".s"))
        {
            const auto assembled = AmdgcnAssemble(src, params);
            WriteFile(assembled, hsaco_file);
        }
        else if(miopen::EndsWith(filename, ".cpp"))
        {
            hsaco_file = HipBuild(dir, filename, src, params, device);
        }
        else
        {
            params += " " + GetCoV3Option(ProduceCoV3());
            WriteFile(src, dir->path / filename);
            dir->Execute(HIP_OC_COMPILER, params + " " + filename + " -o " + hsaco_file.string());
        }
        if(!boost::filesystem::exists(hsaco_file))
            MIOPEN_THROW("Cant find file: " + hsaco_file.string());
    }

#else // MIOPEN_USE_COMGR
    void BuildCodeObjectInMemory(const std::string& params,
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
                comgr::BuildHip(filename, src, params, device, binary);
            else if(miopen::EndsWith(filename, ".s"))
                comgr::BuildAsm(filename, src, params, device, binary);
            else
                comgr::BuildOcl(filename, src, params, device, binary);
        }
        if(binary.empty())
            MIOPEN_THROW("Code object build failed. Source: " + filename);
    }
#endif // MIOPEN_USE_COMGR

    void BuildCodeObject(std::string params, bool is_kernel_str, const std::string& kernel_src)
    {
        std::string filename = is_kernel_str ? "tinygemm.cl" // Fixed name for miopengemm.
                                             : program;
        const std::string src =
            !kernel_src.empty() ? kernel_src : is_kernel_str ? program : GetKernelSrc(program);

        if(miopen::EndsWith(filename, ".cpp"))
        {
            params += " -Wno-everything";
        }
        else if(miopen::EndsWith(filename, ".cl"))
        {
#if MIOPEN_BUILD_DEV
            params += " -Werror" + OclKernelWarningsString();
#else
            params += " -Wno-everything";
#endif
        }
#if MIOPEN_USE_COMGR /// \todo Refactor when functionality stabilize.
        BuildCodeObjectInMemory(params, src, filename);
#else
        BuildCodeObjectInFile(params, src, filename);
#endif
    }
};

HIPOCProgram::HIPOCProgram() {}
HIPOCProgram::HIPOCProgram(const std::string& program_name,
                           std::string params,
                           bool is_kernel_str,
                           std::string dev_name,
                           const std::string& kernel_src)
    : impl(std::make_shared<HIPOCProgramImpl>(
          program_name, params, is_kernel_str, dev_name, kernel_src))
{
}

HIPOCProgram::HIPOCProgram(const std::string& program_name, const boost::filesystem::path& hsaco)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, hsaco))
{
}

HIPOCProgram::HIPOCProgram(const std::string& program_name, const std::string& hsaco)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, hsaco))
{
}

hipModule_t HIPOCProgram::GetModule() const { return impl->module.get(); }

boost::filesystem::path HIPOCProgram::GetCodeObjectPathname() const { return impl->hsaco_file; }

std::string HIPOCProgram::GetCodeObjectBlob() const
{
    return {impl->binary.data(), impl->binary.size()};
}

bool HIPOCProgram::IsCodeObjectInMemory() const { return !impl->binary.empty(); };

} // namespace miopen
