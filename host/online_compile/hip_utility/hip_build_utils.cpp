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

#include <hip_build_utils.hpp>
#include <stringutils.hpp>
#include <tmp_dir.hpp>
#include <env.hpp>
#include <target_properties.hpp>
#include <write_file.hpp>
#include <exec_utils.hpp>
#include <logger.hpp>
#include <config.h>
#include <boost/optional.hpp>
#include <sstream>
#include <string>
#include <stdexcept>

#include <iostream>

OLC_DECLARE_ENV_VAR(OLC_DEBUG_HIP_VERBOSE)
OLC_DECLARE_ENV_VAR(OLC_DEBUG_HIP_DUMP)

#define OLC_HIP_COMPILER "/opt/rocm/llvm/bin/clang++"

namespace online_compile {

bool IsHccCompiler()
{
    static const auto isHcc = EndsWith(OLC_HIP_COMPILER, "hcc");
    return isHcc;
}

bool IsHipClangCompiler()
{
    static const auto isClangXX = EndsWith(OLC_HIP_COMPILER, "clang++");
    return isClangXX;
}

namespace {

inline bool ProduceCoV3()
{
    // Otherwise, let's enable CO v3 for HIP kernels since ROCm 3.0.
    return (HipCompilerVersion() >= external_tool_version_t{3, 0, -1});
}

/// Returns option for enabling/disabling CO v3 generation for the compiler
/// that builds HIP kernels, depending on compiler version etc.
inline const std::string& GetCoV3Option(const bool enable)
{
    /// \note PR #2166 uses the "--hcc-cov3" option when isHCC is true.
    /// It's unclear why... HCC included in ROCm 2.8 does not support it,
    /// perhaps it suits for some older HCC?
    ///
    /// These options are Ok for ROCm 3.0:
    static const std::string option_enable{"-mcode-object-v3"};
    static const std::string no_option{};
    if(enable)
        return option_enable;
    else
        return no_option;
}
} // namespace

static boost::filesystem::path HipBuildImpl(boost::optional<TmpDir>& tmp_dir,
                                            const std::string& filename,
                                            std::string src,
                                            std::string params,
                                            const TargetProperties& target,
                                            const bool testing_mode,
                                            const bool sources_already_reside_on_filesystem)
{
#ifdef __linux__
    // Write out the include files
    // Let's assume includes are overkill for feature tests & optimize'em out.
    if(!testing_mode)
    {
        auto inc_list = GetHipKernelIncList();
        auto inc_path = tmp_dir->path;
        boost::filesystem::create_directories(inc_path);
        for(auto inc_file : inc_list)
        {
            auto inc_src = GetKernelInc(inc_file);
            WriteFile(inc_src, inc_path / inc_file);
        }
    }

    // Sources produced by MLIR-cpp already reside in tmp dir.
    if(!sources_already_reside_on_filesystem)
    {
        src += "\nint main() {}\n";
        WriteFile(src, tmp_dir->path / filename);
    }

    // cppcheck-suppress unreadVariable
    const LcOptionTargetStrings lots(target);

    auto env = std::string("");
    if(IsHccCompiler())
    {
        params += " -amdgpu-target=" + target.Name();
        params += " " + GetCoV3Option(ProduceCoV3());
    }
    else if(IsHipClangCompiler())
    {
        if(params.find("-std=") == std::string::npos)
            params += " --std=c++11";

        if(HipCompilerVersion() < external_tool_version_t{4, 1, 0})
            params += " --cuda-gpu-arch=" + lots.device;
        else
            params += " --cuda-gpu-arch=" + lots.device + lots.xnack;

        params += " --cuda-device-only";
        params += " -c";
        params += " -O3 ";
    }

    params += " -Wno-unused-command-line-argument -I. ";
    params += OLC_STRINGIZE(HIP_COMPILER_FLAGS);
    if(IsHccCompiler())
    {
        env += std::string("KMOPTLLC=\"-mattr=+enable-ds128 ");
        if(HipCompilerVersion() >= external_tool_version_t{2, 8, 0})
            env += " --amdgpu-spill-vgpr-to-agpr=0";
        env += '\"';
    }
    else if(IsHipClangCompiler())
    {
        params += " -mllvm --amdgpu-spill-vgpr-to-agpr=0";
        params += " -mllvm -amdgpu-early-inline-all=true";
        params += " -mllvm -amdgpu-function-calls=false";
    }

    if(online_compile::IsEnabled(OLC_DEBUG_HIP_VERBOSE{}))
    {
        params += " -v";
    }

    if(online_compile::IsEnabled(OLC_DEBUG_HIP_DUMP{}))
    {
        if(IsHccCompiler())
        {
            params += " -gline-tables-only";
            env += " KMDUMPISA=1";
            env += " KMDUMPLLVM=1";
        }
        else if(IsHipClangCompiler())
        {
            params += " -gline-tables-only";
            params += " -save-temps";
        }
    }

    // hip version
    params +=
        std::string(" -DHIP_PACKAGE_VERSION_FLAT=") + std::to_string(HIP_PACKAGE_VERSION_FLAT);

    params += " ";
    auto bin_file = tmp_dir->path / (filename + ".o");

    // compile
    const std::string redirector = testing_mode ? " 1>/dev/null 2>&1" : "";
    tmp_dir->Execute(env + std::string(" ") + OLC_HIP_COMPILER,
                     params + filename + " -o " + bin_file.string() + redirector);
    if(!boost::filesystem::exists(bin_file))
        throw std::runtime_error(filename + " failed to compile");
#ifdef EXTRACTKERNEL_BIN
    if(IsHccCompiler())
    {
        // call extract kernel
        tmp_dir->Execute(EXTRACTKERNEL_BIN, " -i " + bin_file.string());
        auto hsaco =
            std::find_if(boost::filesystem::directory_iterator{tmp_dir->path}, {}, [](auto entry) {
                return (entry.path().extension() == ".hsaco");
            });

        if(hsaco == boost::filesystem::directory_iterator{})
        {
            fdt_log(LogLevel::Info, "HipBuild", "failed to find *.hsaco in ")
                << hsaco->path().string() << std::endl;
        }

        return hsaco->path();
    }
#endif
    return bin_file;
#else
    (void)filename;
    (void)params;
    throw std::runtimer_error("HIP kernels are only supported in Linux");
#endif
}

boost::filesystem::path HipBuild(boost::optional<TmpDir>& tmp_dir,
                                 const std::string& filename,
                                 std::string src,
                                 std::string params,
                                 const TargetProperties& target,
                                 const bool sources_already_reside_on_filesystem)
{
    return HipBuildImpl(
        tmp_dir, filename, src, params, target, false, sources_already_reside_on_filesystem);
}

void bin_file_to_str(const boost::filesystem::path& file, std::string& buf)
{
    std::ifstream bin_file_ptr(file.string().c_str(), std::ios::binary);
    std::ostringstream bin_file_strm;
    bin_file_strm << bin_file_ptr.rdbuf();
    buf = bin_file_strm.str();
}

static external_tool_version_t HipCompilerVersionImpl()
{
    external_tool_version_t version;
    if(IsHccCompiler())
    {
        const std::string path(OLC_HIP_COMPILER);
        const std::string mandatory_prefix("(based on HCC ");
        do
        {
            if(path.empty() || !std::ifstream(path).good())
                break;

            std::stringstream out;
            if(online_compile::exec::Run(path + " --version", nullptr, &out) != 0)
                break;

            std::string line;
            while(!out.eof())
            {
                std::getline(out, line);
                fdt_log() << line;
                auto begin = line.find(mandatory_prefix);
                if(begin == std::string::npos)
                    continue;

                begin += mandatory_prefix.size();
                int v3, v2, v1 = v2 = v3 = -1;
                char c2, c1 = c2 = 'X';
                std::istringstream iss(line.substr(begin));
                iss >> v1 >> c1 >> v2 >> c2 >> v3;
                if(!iss.fail() && v1 >= 0)
                {
                    version.major = v1;
                    if(c1 == '.' && v2 >= 0)
                    {
                        version.minor = v2;
                        if(c2 == '.' && v3 >= 0)
                            version.patch = v3;
                    }
                }
                break;
            }
        } while(false);
    }
    else
    {
#ifdef HIP_PACKAGE_VERSION_MAJOR
        fdt_log(
            LogLevel::Info, "HipCompilerVersion", "Read version information from HIP package...");
        version.major = HIP_PACKAGE_VERSION_MAJOR;
#ifdef HIP_PACKAGE_VERSION_MINOR
        version.minor = HIP_PACKAGE_VERSION_MINOR;
#else
        version.minor = 0;
#endif
#ifdef HIP_PACKAGE_VERSION_PATCH
        version.patch = HIP_PACKAGE_VERSION_PATCH;
#else
        version.patch = 0;
#endif
#else // HIP_PACKAGE_VERSION_MAJOR is not defined. CMake failed to find HIP package.
        fdt_log(LogLevel::Info, "HipCompilerVersion", "...assuming 3.2.0 (hip-clang RC)");
        version.major = 3;
        version.minor = 2;
        version.patch = 0;
#endif
    }
    fdt_log() << version.major << '.' << version.minor << '.' << version.patch << std::endl;
    return version;
}

external_tool_version_t HipCompilerVersion()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static auto once = HipCompilerVersionImpl();
    return once;
}

bool operator>(const external_tool_version_t& lhs, const external_tool_version_t& rhs)
{
    if(lhs.major > rhs.major)
        return true;
    else if(lhs.major == rhs.major)
    {
        if(lhs.minor > rhs.minor)
            return true;
        else if(lhs.minor == rhs.minor)
            return (lhs.patch > rhs.patch);
        else
            return false;
    }
    else
        return false;
}

bool operator<(const external_tool_version_t& lhs, const external_tool_version_t& rhs)
{
    return rhs > lhs;
}
bool operator>=(const external_tool_version_t& lhs, const external_tool_version_t& rhs)
{
    return !(lhs < rhs);
}

bool operator<=(const external_tool_version_t& lhs, const external_tool_version_t& rhs)
{
    return !(lhs > rhs);
}

} // namespace online_compile
