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
#ifndef GUARD_OLC_HIPOC_PROGRAM_IMPL_HPP
#define GUARD_OLC_HIPOC_PROGRAM_IMPL_HPP

#include <target_properties.hpp>
#include <manage_ptr.hpp>
#include <tmp_dir.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/optional.hpp>
#include <hip/hip_runtime_api.h>

namespace olCompile {

using hipModulePtr = OLC_MANAGE_PTR(hipModule_t, hipModuleUnload);

struct HIPOCProgramImpl
{
    HIPOCProgramImpl(){};
    HIPOCProgramImpl(const std::string& program_name, const boost::filesystem::path& filespec);

    HIPOCProgramImpl(const std::string& program_name,
                     std::string params,
                     const TargetProperties& target_);

    std::string program;
    TargetProperties target;
    boost::filesystem::path hsaco_file;
    hipModulePtr module;
    boost::optional<TmpDir> dir;
    std::vector<char> binary;

    void
    BuildCodeObjectInFile(std::string& params, const std::string& src, const std::string& filename);
    void BuildCodeObject(std::string params);
};
} // namespace olCompile
#endif // GUARD_OLC_HIPOC_PROGRAM_IMPL_HPP
