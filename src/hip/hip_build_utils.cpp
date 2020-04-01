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

#include <miopen/hip_build_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/exec_utils.hpp>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <boost/optional.hpp>
#include <sstream>
#include <string>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_HIP_ENFORCE_COV3)

#define WORKAROUND_ISSUE_2514 1

namespace miopen {

namespace {

inline bool IsHccCompiler()
{
    static const auto isHcc = EndsWith(MIOPEN_HIP_COMPILER, "hcc");
    return isHcc;
}

inline bool ProduceCoV3()
{
    // If env.var is set, then let's follow it.
    if(IsEnabled(MIOPEN_DEBUG_HIP_ENFORCE_COV3{}))
        return true;
    if(IsDisabled(MIOPEN_DEBUG_HIP_ENFORCE_COV3{}))
        return false;
    // Otherwise, let's enable CO v3 for HIP kernels since ROCm 3.0.
    return (HipGetHccVersion() >= external_tool_version_t{3, 0, -1});
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

boost::filesystem::path HipBuild(boost::optional<TmpDir>& tmp_dir,
                                 const std::string& filename,
                                 std::string src,
                                 std::string params,
                                 const std::string& dev_name)
{
#ifdef __linux__
    // write out the include files
    auto inc_list = GetKernelIncList();
    auto inc_path = tmp_dir->path;
    boost::filesystem::create_directories(inc_path);
    for(auto inc_file : inc_list)
    {
        auto inc_src = GetKernelInc(inc_file);
        WriteFile(inc_src, inc_path / inc_file);
    }
    src += "\nint main() {}\n";
    WriteFile(src, tmp_dir->path / filename);
    if(IsHccCompiler())
    {
        params += " -amdgpu-target=" + dev_name;
    }
    else
    {
        if(params.find("-std=") == std::string::npos)
            params += " --std=c++11";
        params += " --cuda-gpu-arch=" + dev_name;
        params += " --cuda-device-only -c";
        params += " -O3 ";
    }
    params += " " + GetCoV3Option(ProduceCoV3());

    // params += " -Wno-unused-command-line-argument -c -fno-gpu-rdc -I. ";
    params += " -Wno-unused-command-line-argument -I. ";
    params += MIOPEN_STRINGIZE(HIP_COMPILER_FLAGS);
    params += " ";
    auto bin_file = tmp_dir->path / (filename + ".o");
    // compile
    auto env = std::string("KMOPTLLC=\"-mattr=+enable-ds128 -amdgpu-enable-global-sgpr-addr");
    if(miopen::HipGetHccVersion() >= external_tool_version_t{2, 8, 0})
        env += " --amdgpu-spill-vgpr-to-agpr=0";
    env += '\"';
    tmp_dir->Execute(env + std::string(" ") + MIOPEN_HIP_COMPILER,
                     params + filename + " -o " + bin_file.string());
    if(!boost::filesystem::exists(bin_file))
        MIOPEN_THROW(filename + " failed to compile");
#ifdef EXTRACTKERNEL_BIN
    if(IsHccCompiler())
    {
        // call extract kernel
        tmp_dir->Execute(EXTRACTKERNEL_BIN, " -i " + bin_file.string());
        auto hsaco =
            std::find_if(boost::filesystem::directory_iterator{tmp_dir->path},
                         {},
                         [](auto entry) { return (entry.path().extension() == ".hsaco"); });

        if(hsaco == boost::filesystem::directory_iterator{})
        {
            MIOPEN_LOG_E("failed to find *.hsaco in " << hsaco->path().string());
        }

        return hsaco->path();
    }
    else
#endif
    {
        return bin_file;
    }
#else
    (void)filename;
    (void)params;
    MIOPEN_THROW("HIP kernels are only supported in Linux");
#endif
}

void bin_file_to_str(const boost::filesystem::path& file, std::string& buf)
{
    std::ifstream bin_file_ptr(file.string().c_str(), std::ios::binary);
    std::ostringstream bin_file_strm;
    bin_file_strm << bin_file_ptr.rdbuf();
    buf = bin_file_strm.str();
}

static external_tool_version_t HipGetHccVersionImpl()
{
    external_tool_version_t hcc_version;
    const std::string path(MIOPEN_HIP_COMPILER);
    const std::string mandatory_prefix("(based on HCC ");
    do
    {
        if(path.empty() || !std::ifstream(path).good())
            break;

        std::stringstream out;
        MIOPEN_LOG_NQI2("Running: " << '\'' << path << " --version" << '\'');
        if(miopen::exec::Run(path + " --version", nullptr, &out) != 0)
            break;

#if WORKAROUND_ISSUE_2514
        // If compiler is not hcc and mandatory prefix is not found,
        // then assume hip-clang 3.2.0.
        bool mandatory_prefix_found = false;
#endif
        std::string line;
        while(!out.eof())
        {
            std::getline(out, line);
            MIOPEN_LOG_NQI2(line);
            auto begin = line.find(mandatory_prefix);
            if(begin == std::string::npos)
                continue;

#if WORKAROUND_ISSUE_2514
            mandatory_prefix_found = true;
#endif
            begin += mandatory_prefix.size();
            int v3, v2, v1 = v2 = v3 = -1;
            char c2, c1 = c2 = 'X';
            std::istringstream iss(line.substr(begin));
            iss >> v1 >> c1 >> v2 >> c2 >> v3;
            if(!iss.fail() && v1 >= 0)
            {
                hcc_version.major = v1;
                if(c1 == '.' && v2 >= 0)
                {
                    hcc_version.minor = v2;
                    if(c2 == '.' && v3 >= 0)
                        hcc_version.patch = v3;
                }
            }
            break;
        }
#if WORKAROUND_ISSUE_2514
        if(!mandatory_prefix_found && !IsHccCompiler())
        {
            MIOPEN_LOG_NQI2("Assuming 3.2.0 (hip-clang?)");
            hcc_version.major = 3;
            hcc_version.minor = 2;
            hcc_version.patch = 0;
        }
#endif
    } while(false);
    MIOPEN_LOG_NQI("HCC base: " << hcc_version.major << '.' << hcc_version.minor << '.'
                                << hcc_version.patch);
    return hcc_version;
}

external_tool_version_t HipGetHccVersion()
{
    static auto once = HipGetHccVersionImpl();
    return once;
}

bool external_tool_version_t::operator>=(const external_tool_version_t& rhs) const
{
    if(major > rhs.major)
        return true;
    else if(major == rhs.major)
    {
        if(minor > rhs.minor)
            return true;
        else if(minor == rhs.minor)
            return (patch >= rhs.patch);
        else
            return false;
    }
    else
        return false;
}

} // namespace miopen
