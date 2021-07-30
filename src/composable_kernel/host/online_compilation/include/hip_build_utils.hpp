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
#ifndef OLC_GUARD_OLC_HIP_BUILD_UTILS_HPP
#define OLC_GUARD_OLC_HIP_BUILD_UTILS_HPP

#include <target_properties.hpp>
#include <kernel.hpp>
#include <boost/optional.hpp>
#include <string>

namespace olCompile {

boost::filesystem::path HipBuild(boost::optional<olCompile::TmpDir>& tmp_dir,
                                 const std::string& filename,
                                 std::string src,
                                 std::string params,
                                 const TargetProperties& target,
                                 bool sources_already_reside_on_filesystem = false);

void bin_file_to_str(const boost::filesystem::path& file, std::string& buf);

struct external_tool_version_t
{
    int major = -1;
    int minor = -1;
    int patch = -1;
    friend bool operator>(const external_tool_version_t& lhs, const external_tool_version_t& rhs);
    friend bool operator<(const external_tool_version_t& lhs, const external_tool_version_t& rhs);
    friend bool operator>=(const external_tool_version_t& lhs, const external_tool_version_t& rhs);
    friend bool operator<=(const external_tool_version_t& lhs, const external_tool_version_t& rhs);
};

external_tool_version_t HipCompilerVersion();

bool IsHccCompiler();
bool IsHipClangCompiler();

class LcOptionTargetStrings
{
    public:
    const std::string& device;
    const std::string xnack;

    private:
    const std::string sramecc;
    const std::string sramecc_reported;

    public:
    const std::string targetId;
    LcOptionTargetStrings(const TargetProperties& target)
        : device(target.Name()),
          xnack([&]() -> std::string {
              if(target.Xnack())
                  return std::string{":xnack"} + (*target.Xnack() ? "+" : "-");
              return {};
          }()),
          sramecc([&]() -> std::string {
              if(target.Sramecc())
                  return std::string{":sramecc"} + (*target.Sramecc() ? "+" : "-");
              return {};
          }()),
          sramecc_reported([&]() -> std::string {
              if(target.SrameccReported())
                  return std::string{":sramecc"} + (*target.SrameccReported() ? "+" : "-");
              return {};
          }()),
          targetId(device + sramecc + xnack)
    {
    }
};

} // namespace olCompile

#endif
