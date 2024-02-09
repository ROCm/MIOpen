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

#include <miopen/config.h>
#include <miopen/hip_build_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/exec_utils.hpp>
#include <miopen/logger.hpp>
#include <miopen/env.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/target_properties.hpp>
#include <boost/optional.hpp>
#include <sstream>
#include <string>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_HIP_VERBOSE)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_HIP_DUMP)

#if MIOPEN_USE_COMPILER_PATHS_V2

// List to hold Relative Directory Path
// for each Compiler Flags from ROCM Install Path
// The definition of this variable expected to be available in C/C++
// code of the Target Library.
static const char* MIOPEN_CMP_FLAG_REL_PATHS[MIOPEN_CMP_MAX_Flag] = {
    "llvm/bin/clang", "bin/clang-ocl", "llvm/bin/clang++", "llvm/bin/clang-offload-bundler"};

// Global variable List to hold the Compiler Flag Paths.
char MIOPEN_CMP_FLAGS[MIOPEN_CMP_MAX_Flag][PATH_MAX_LEN] = {0};

// Function to generate the MIOPEN Compiler Flag Value using
// ROCm Base Install Path fetched using getROCmInstallPath()
// This approach depends on the getROCmInstallPath() provided by rocm-core
// This flag Paths are expected to be deprecated/modified in upcoming MAJOR Releases.
char* generateFlagValue(MIOPEN_CMP_Flags_t x)
{
    char* flagPath = (char*)0;
    unsigned int len = 0;
    if(MIOPEN_CMP_MAX_Flag <= x)
    {
        return (char*)0;
    }
    if(0 == MIOPEN_CMP_FLAGS[x][0])
    {
        PathErrors_t ret = getROCmInstallPath(&flagPath, &len);
        if(PathSuccess == ret)
        {
            if(PATH_MAX_LEN > (len + strlen(MIOPEN_CMP_FLAG_REL_PATHS[x])))
            {
                sprintf(
                    (char*)&(MIOPEN_CMP_FLAGS[x]), "%s%s", flagPath, MIOPEN_CMP_FLAG_REL_PATHS[x]);
            }
            free(flagPath);
        }
    }
    return (char*)&(MIOPEN_CMP_FLAGS[x]);
}

#endif // MIOPEN_USE_COMPILER_PATHS_V2

namespace miopen {

static fs::path HipBuildImpl(boost::optional<TmpDir>& tmp_dir,
                             const std::string& filename,
                             std::string src,
                             std::string params,
                             const TargetProperties& target,
                             const bool testing_mode)
{
#ifdef __linux__
    // Write out the include files
    // Let's assume includes are overkill for feature tests & optimize'em out.
    if(!testing_mode)
    {
        auto inc_list = GetHipKernelIncList();
        auto inc_path = tmp_dir->path;
        fs::create_directories(inc_path);
        for(const auto& inc_file : inc_list)
        {
            auto inc_src = GetKernelInc(inc_file);
            WriteFile(inc_src, inc_path / inc_file);
        }
    }

    src += "\nint main() {}\n";
    WriteFile(src, tmp_dir->path / filename);

    // cppcheck-suppress unreadVariable
    const LcOptionTargetStrings lots(target);

    auto env = std::string("");

    if(params.find("-std=") == std::string::npos)
        params += " --std=c++17";

#if HIP_PACKAGE_VERSION_FLAT >= 6001024000ULL
    size_t pos = 0;
    while((pos = params.find("-mcpu=", pos)) != std::string::npos)
    {
        size_t endpos = params.find(' ', pos);
        if(endpos == std::string::npos)
        {
            params.erase(pos, std::string::npos);
            break;
        }
        params.erase(pos, endpos - pos);
    }
#endif

    params += " --cuda-gpu-arch=" + lots.device + lots.xnack;
    params += " --cuda-device-only";
    params += " -c";
    params += " -O3 ";
    params += " -Wno-unused-command-line-argument -I. ";
    params += MIOPEN_STRINGIZE(HIP_COMPILER_FLAGS);

#if MIOPEN_BUILD_DEV
    if(miopen::IsEnabled(ENV(MIOPEN_DEBUG_HIP_VERBOSE)))
    {
        params += " -v";
    }
    if(miopen::IsEnabled(ENV(MIOPEN_DEBUG_HIP_DUMP)))
    {
        params += " -gline-tables-only";
        params += " -save-temps";
    }
#endif

    // hip version
    params +=
        std::string(" -DHIP_PACKAGE_VERSION_FLAT=") + std::to_string(HIP_PACKAGE_VERSION_FLAT);

    params += " ";
    auto bin_file = tmp_dir->path / (filename + ".o");

    // compile
    {
        const std::string redirector = testing_mode ? " 1>/dev/null 2>&1" : "";
        const std::string cmd        = env + std::string(" ") + MIOPEN_HIP_COMPILER;
        const std::string args       = params + filename + " -o " + bin_file.string() + redirector;
        tmp_dir->Execute(cmd, args);
        if(!fs::exists(bin_file))
            MIOPEN_THROW("Failed cmd: '" + cmd + "', args: '" + args + '\'');
    }

#if defined(MIOPEN_OFFLOADBUNDLER_BIN) && !MIOPEN_BACKEND_HIP
    // Unbundling is not required for HIP runtime && hip-clang
    tmp_dir->Execute(MIOPEN_OFFLOADBUNDLER_BIN,
                     "--type=o "
                     "--targets=hipv4-amdgcn-amd-amdhsa-" +
                         (std::string{'-'} + lots.device + lots.xnack) +
                         " --inputs=" + bin_file.string() + " --outputs=" + bin_file.string() +
                         ".hsaco --unbundle");

    auto hsaco = std::find_if(fs::directory_iterator{tmp_dir->path}, {}, [](auto entry) {
        return (entry.path().extension() == ".hsaco");
    });

    if(hsaco == fs::directory_iterator{})
    {
        MIOPEN_LOG_E("failed to find *.hsaco in " << hsaco->path().string());
    }
    return hsaco->path();
#endif
    return bin_file;
#else
    (void)filename;
    (void)params;
    MIOPEN_THROW("HIP kernels are only supported in Linux");
#endif
}

fs::path HipBuild(boost::optional<TmpDir>& tmp_dir,
                  const std::string& filename,
                  std::string src,
                  std::string params,
                  const TargetProperties& target)
{
    if(miopen::solver::support_amd_buffer_atomic_fadd(target.Name()))
        params += " -DCK_AMD_BUFFER_ATOMIC_FADD_RETURNS_FLOAT=1";
    return HipBuildImpl(tmp_dir, filename, src, params, target, false);
}

void bin_file_to_str(const fs::path& file, std::string& buf)
{
    std::ifstream bin_file_ptr(file.string().c_str(), std::ios::binary);
    std::ostringstream bin_file_strm;
    bin_file_strm << bin_file_ptr.rdbuf();
    buf = bin_file_strm.str();
}

} // namespace miopen
