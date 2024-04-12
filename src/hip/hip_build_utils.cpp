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

#if MIOPEN_OFFLINE_COMPILER_PATHS_V2

// Include rocm-core header for get ROCm Install Path Method
#include <rocm-core/rocm_getpath.h>

// Flags to hold Relative Directory Path
// for each Compiler Flags from ROCM Install Path
// This flag Paths are expected to be deprecated/modified
// in upcoming MAJOR Releases.
#define MIOPEN_CLANG_REL_PATH "llvm/bin/clang"
#define MIOPEN_OCL_REL_PATH "bin/clang"
#define MIOPEN_CPPCLANG_REL_PATH "llvm/bin/clang++"
#define MIOPEN_OFFLOADBUNDLER_REL_PATH "llvm/bin/clang-offload-bundler"

// Function to generate the MIOPEN Compiler Path Value using
// ROCm Base Install Path fetched using getROCmInstallPath()
// This approach depends on the getROCmInstallPath() provided by rocm-core
// This flag Paths are expected to be deprecated/modified in upcoming MAJOR Releases.
static std::string generateCompilerPathValue(const char* relativePath)
{
    char* rocmPath   = nullptr;
    unsigned int len = 0;
    std::string compilerPathValue;
    if(nullptr != relativePath)
    {
        PathErrors_t ret = getROCmInstallPath(&rocmPath, &len);
        if(PathSuccess == ret)
        {
            compilerPathValue = std::string(rocmPath) + std::string(relativePath);
            // Free rocmPath memory returned (allocated by getROCmInstallPath())
            free(rocmPath);
        }
    }
    return compilerPathValue;
}

// API to get MIOPEN AMD GCN Assembler Path Values.
const char* getAMDGCNAssemblerPath()
{
    static const std::string path = generateCompilerPathValue(MIOPEN_CLANG_REL_PATH);
    return path.c_str();
}

// API to get MIOPEN OpenCL Compiler Path Values.
const char* getOpenCLCompilerPath()
{
    static const std::string path = generateCompilerPathValue(MIOPEN_OCL_REL_PATH);
    return path.c_str();
}

// API to get MIOPEN HIP Compiler Path Values.
const char* getHIPCompilerPath()
{
    static const std::string path = generateCompilerPathValue(MIOPEN_CPPCLANG_REL_PATH);
    return path.c_str();
}

// API to get MIOPEN Compiler Offload Bundler bin Path Values.
const char* getOffloadBundlerBinPath()
{
    static const std::string path = generateCompilerPathValue(MIOPEN_OFFLOADBUNDLER_REL_PATH);
    return path.c_str();
}

#endif // MIOPEN_OFFLINE_COMPILER_PATHS_V2

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
    auto bin_file = make_object_file_name(tmp_dir.get() / filename);

    // compile
    {
        const std::string redirector = testing_mode ? " 1>/dev/null 2>&1" : "";
        const std::string cmd        = env + std::string(" ") + MIOPEN_HIP_COMPILER;
        const std::string args       = params + filename + " -o " + bin_file + redirector;
        tmp_dir->Execute(cmd, args);
        if(!fs::exists(bin_file))
            MIOPEN_THROW("Failed cmd: '" + cmd + "', args: '" + args + '\'');
    }

#if defined(MIOPEN_OFFLOADBUNDLER_BIN) && !MIOPEN_BACKEND_HIP
    // Unbundling is not required for HIP runtime && hip-clang
    tmp_dir->Execute(MIOPEN_OFFLOADBUNDLER_BIN,
                     "--type=o "
                     "--targets=hipv4-amdgcn-amd-amdhsa-" +
                         (std::string{'-'} + lots.device + lots.xnack) + " --inputs=" + bin_file +
                         " --outputs=" + bin_file + ".hsaco --unbundle");

    auto hsaco = std::find_if(fs::directory_iterator{tmp_dir->path}, {}, [](auto entry) {
        return (entry.path().extension() == ".hsaco");
    });

    if(hsaco == fs::directory_iterator{})
    {
        MIOPEN_LOG_E("failed to find *.hsaco in " << hsaco->path());
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

} // namespace miopen
