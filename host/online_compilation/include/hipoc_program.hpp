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
#ifndef GUARD_OLC_HIPOC_PROGRAM_HPP
#define GUARD_OLC_HIPOC_PROGRAM_HPP

#include <target_properties.hpp>
#include <manage_ptr.hpp>
#include <hipoc_program_impl.hpp>
#include <boost/filesystem/path.hpp>
#include <hip/hip_runtime_api.h>
#include <string>

namespace olCompile {

struct HIPOCProgramImpl;
struct HIPOCProgram
{
    HIPOCProgram();
    /// This ctor builds the program from source, initializes module.
    /// Also either CO pathname (typically if offline tools were used)
    /// or binary blob (if comgr was used to build the program)
    /// is initialized. GetModule(), GetCodeObjectPathname(),
    /// GetCodeObjectBlob() return appropriate data after this ctor.
    /// Other ctors only guarantee to initialize module.
    HIPOCProgram(const std::string& program_name,
                 std::string params,
                 const TargetProperties& target);
    HIPOCProgram(const std::string& program_name, const boost::filesystem::path& hsaco);
    std::shared_ptr<const HIPOCProgramImpl> impl;
    hipModule_t GetModule() const;
    /// \return Pathname of CO file, if it resides on the filesystem.
    boost::filesystem::path GetCodeObjectPathname() const;
    /// \return Copy of in-memory CO blob.
    std::string GetCodeObjectBlob() const;
    /// \return True if CO blob resides in-memory.
    /// False if CO resides on filesystem.
    bool IsCodeObjectInMemory() const;
};
} // namespace olCompile

#endif
