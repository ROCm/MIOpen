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
#include <hip_build_utils.hpp>
#include <hipoc_program.hpp>
#include <kernel.hpp>
#include <stringutils.hpp>
#include <target_properties.hpp>
#include <env.hpp>
#include <write_file.hpp>
#include <boost/optional.hpp>
#include <boost/filesystem/operations.hpp>

#include <cstring>
#include <mutex>
#include <sstream>

#include <unistd.h>

namespace online_compile {

static hipModulePtr CreateModule(const boost::filesystem::path& hsaco_file)
{
    hipModule_t raw_m;
    MY_HIP_CHECK(hipModuleLoad(&raw_m, hsaco_file.string().c_str()));
    hipModulePtr m{raw_m};
    return m;
}

template <typename T> /// intended for std::string and std::vector<char>
hipModulePtr CreateModuleInMem(const T& blob)
{
    hipModule_t raw_m;
    MY_HIP_CHECK(hipModuleLoadData(&raw_m, reinterpret_cast<const void*>(blob.data())));
    hipModulePtr m{raw_m};
    return m;
}

HIPOCProgramImpl::HIPOCProgramImpl(const std::string& program_name,
                                   const boost::filesystem::path& filespec)
    : program(program_name), hsaco_file(filespec)
{
    this->module = CreateModule(hsaco_file);
}

HIPOCProgramImpl::HIPOCProgramImpl(const std::string& program_name,
                                   std::string params,
                                   const TargetProperties& target_)
    : program(program_name), target(target_)
{
    BuildCodeObject(params);
    if(!binary.empty())
    {
        module = CreateModuleInMem(this->binary);
    }
    else
    {
        module = CreateModule(this->hsaco_file);
    }
}

void HIPOCProgramImpl::BuildCodeObjectInFile(std::string& params,
                                             const std::string& src,
                                             const std::string& filename)
{

    this->dir.emplace(filename);
    hsaco_file = dir->path / (filename + ".o");

    if(online_compile::EndsWith(filename, ".cpp"))
    {
        hsaco_file = HipBuild(dir, filename, src, params, target);
    }
    else
        throw std::runtime_error("Only HIP kernel source of .cpp file is supported");

    if(!boost::filesystem::exists(hsaco_file))
        throw std::runtime_error("Cant find file: " + hsaco_file.string());
}

void HIPOCProgramImpl::BuildCodeObject(std::string params)
{
    std::string filename = program;

    if(online_compile::EndsWith(filename, ".cpp"))
    {
        params += " -Wno-everything";
    }

    BuildCodeObjectInFile(params, GetKernelSrc(this->program), filename);
}

HIPOCProgram::HIPOCProgram() {}
HIPOCProgram::HIPOCProgram(const std::string& program_name,
                           std::string params,
                           const TargetProperties& target)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, params, target))
{
}

HIPOCProgram::HIPOCProgram(const std::string& program_name, const boost::filesystem::path& hsaco)
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

} // namespace online_compile
