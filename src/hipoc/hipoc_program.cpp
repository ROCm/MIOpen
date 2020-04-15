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
#include <boost/optional.hpp>
#include <sstream>

#include <unistd.h>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_OPENCL_ENFORCE_COV3)

#define MIOPEN_WORKAROUND_SWDEV_225285 1

namespace miopen {

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
    return (HipGetHccVersion() >= external_tool_version_t{3, 0, -1});
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

hipModulePtr CreateModule(const boost::filesystem::path& hsaco_file)
{
    hipModule_t raw_m;
    auto status = hipModuleLoad(&raw_m, hsaco_file.string().c_str());
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed creating module");
    return m;
}

hipModulePtr CreateModuleInMem(const std::string& hsaco)
{
    hipModule_t raw_m;
    auto status = hipModuleLoadData(&raw_m, reinterpret_cast<const void*>(hsaco.data()));
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed loading module");
    return m;
}

struct HIPOCProgramImpl
{
    HIPOCProgramImpl(const std::string& program_name, const boost::filesystem::path& hsaco)
        : name(program_name), hsaco_file(hsaco)
    {
        this->module = CreateModule(this->hsaco_file);
    }
    HIPOCProgramImpl(const std::string& program_name,
                     const std::string& hsaco) // the actual module and not just the filename
#if MIOPEN_WORKAROUND_SWDEV_225285
        : name(program_name)
#else
        : name(program_name),
          module(CreateModuleInMem(hsaco))
#endif
    {
#if MIOPEN_WORKAROUND_SWDEV_225285
        // This should use the CreateModuleInMem function above, however that is disabled
        // due to SWDEV-225285
        TmpDir tmp_dir("miopen");
        auto file_path =
            tmp_dir.path / boost::filesystem::unique_path("miopen-%%%%-%%%%-%%%%-%%%%");
        WriteFile(hsaco, file_path);
        this->module = CreateModule(file_path);
#endif
    }
    HIPOCProgramImpl(const std::string& program_name,
                     std::string params,
                     bool is_kernel_str,
                     std::string _dev_name,
                     const std::string& kernel_src)
        : name(program_name), dev_name(_dev_name)
    {
        this->BuildModule(program_name, params, is_kernel_str, kernel_src);
        this->module = CreateModule(this->hsaco_file);
    }
    std::string name;
    std::string dev_name;
    boost::filesystem::path hsaco_file;
    hipModulePtr module;
    boost::optional<TmpDir> dir;
    void BuildModule(const std::string& program_name,
                     std::string params,
                     bool is_kernel_str,
                     const std::string& kernel_src)
    {
        std::string filename =
            is_kernel_str ? "tinygemm.cl" : program_name; // jn : don't know what this is
        dir.emplace(filename);

        hsaco_file = dir->path / (filename + ".o");
        std::string src;
        if(kernel_src.empty())
            src = is_kernel_str ? program_name : GetKernelSrc(program_name);
        else
            src = kernel_src;
        if(!is_kernel_str && miopen::EndsWith(program_name, ".so"))
        {
            WriteFile(src, hsaco_file);
        }
        else if(!is_kernel_str && miopen::EndsWith(program_name, ".s"))
        {
            AmdgcnAssemble(src, params);
            WriteFile(src, hsaco_file);
        }
        else if(!is_kernel_str && miopen::EndsWith(program_name, ".cpp"))
        {
#if MIOPEN_BUILD_DEV
            params += " -Werror" + HipKernelWarningsString();
#else
            params += " -Wno-everything";
#endif
            hsaco_file = HipBuild(dir, filename, src, params, dev_name);
        }
        else
        {

            WriteFile(src, dir->path / filename);

#if MIOPEN_BUILD_DEV
            params += " -Werror" + OclKernelWarningsString();
#else
            params += " -Wno-everything";
#endif
            params += " " + GetCoV3Option(ProduceCoV3());
            dir->Execute(HIP_OC_COMPILER, params + " " + filename + " -o " + hsaco_file.string());
        }
        if(!boost::filesystem::exists(hsaco_file))
            MIOPEN_THROW("Cant find file: " + hsaco_file.string());
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

hipModule_t HIPOCProgram::GetModule() const { return this->impl->module.get(); }

boost::filesystem::path HIPOCProgram::GetBinary() const { return this->impl->hsaco_file; }

} // namespace miopen
