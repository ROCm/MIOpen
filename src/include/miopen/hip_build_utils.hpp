/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_GUARD_MLOPEN_HIP_BUILD_UTILS_HPP
#define MIOPEN_GUARD_MLOPEN_HIP_BUILD_UTILS_HPP

#include <miopen/config.h>
#include <miopen/kernel.hpp>
#include <miopen/tmp_dir.hpp>
#include <miopen/write_file.hpp>
#include <boost/optional.hpp>
#include <string>

#ifndef HIP_PACKAGE_VERSION_MAJOR
#define HIP_PACKAGE_VERSION_MAJOR 0
#endif
#ifndef HIP_PACKAGE_VERSION_MINOR
#define HIP_PACKAGE_VERSION_MINOR 0
#endif
#ifndef HIP_PACKAGE_VERSION_PATCH
#define HIP_PACKAGE_VERSION_PATCH 0
#endif

// 3 decimal digits for major and minor, 6 digits for patch number.
// Max number is 999,999,999999 == 0xE8,D4A5,0FFF that fits into 64-bit math.
#if HIP_PACKAGE_VERSION_MAJOR > 999 || HIP_PACKAGE_VERSION_MAJOR > 999 || \
    HIP_PACKAGE_VERSION_PATCH > 999999
#error "Too big HIP version number(s)"
#endif
#define HIP_PACKAGE_VERSION_FLAT                                                   \
    ((HIP_PACKAGE_VERSION_MAJOR * 1000ULL + HIP_PACKAGE_VERSION_MINOR) * 1000000 + \
     HIP_PACKAGE_VERSION_PATCH)

namespace miopen {

boost::filesystem::path HipBuild(boost::optional<miopen::TmpDir>& tmp_dir,
                                 const std::string& filename,
                                 std::string src,
                                 std::string params,
                                 const std::string& dev_name);

void bin_file_to_str(const boost::filesystem::path& file, std::string& buf);

struct external_tool_version_t
{
    int major = -1;
    int minor = -1;
    int patch = -1;
    bool operator>(const external_tool_version_t& rhs) const;
    bool operator>=(const external_tool_version_t& rhs) const;
    bool operator<(const external_tool_version_t& rhs) const;
};

external_tool_version_t HipCompilerVersion();

bool IsHccCompiler();
bool IsHipClangCompiler();

} // namespace miopen

#endif
