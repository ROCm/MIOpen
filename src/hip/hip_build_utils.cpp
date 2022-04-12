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
#include <miopen/rocm_features.hpp>
#include <miopen/solver/implicitgemm_util.hpp>
#include <miopen/target_properties.hpp>
#include <boost/optional.hpp>
#include <sstream>
#include <string>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_HIP_VERBOSE)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_HIP_DUMP)

namespace miopen {

namespace {

inline void InplaceRemoveAllTokensThatInclude(std::string& str,
                                              const std::string& search_1,
                                              const std::string& search_2)
{
    for(std::size_t pos = 0;;)
    {
        if((pos = str.find(search_1, pos)) == std::string::npos)
            break;

        auto beg       = str.find_last_of(" \t\n\r", pos);
        beg            = (beg == std::string::npos) ? 0 : beg + 1;
        const auto end = str.find_first_of(" \t\n\r", pos);

        auto len  = std::string::npos;
        auto next = std::string::npos;
        if(end != std::string::npos)
        {
            next = str.find_first_not_of(" \t\n\r", end);
            if(next != std::string::npos)
                len = next - beg;
        }

        const auto tok =
            end == std::string::npos ? std::string{str, beg} : std::string{str, beg, end - beg};
        if(tok.find(search_2) != std::string::npos)
        {
            str.erase(beg, len);
            if(len == std::string::npos)
                break;
            pos = beg;
        }
        else
        {
            pos = next;
        }
    }
}

} // namespace

static boost::filesystem::path HipBuildImpl(boost::optional<TmpDir>& tmp_dir,
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
        boost::filesystem::create_directories(inc_path);
        for(auto inc_file : inc_list)
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
        params += " --std=c++11";

#if HIP_PACKAGE_VERSION_FLAT < 4001000000ULL
    params += " --cuda-gpu-arch=" + lots.device;
#else
    params += " --cuda-gpu-arch=" + lots.device + lots.xnack;
#endif
    params += " --cuda-device-only";
    params += " -c";
    params += " -O3 ";

    params += " -Wno-unused-command-line-argument -I. ";

    {
        std::string hip_compiler_flags = MIOPEN_STRINGIZE(HIP_COMPILER_FLAGS);
        InplaceRemoveAllTokensThatInclude(hip_compiler_flags, "clang_rt.builtins", "x86_64");
        params += hip_compiler_flags;
    }

#if HIP_PACKAGE_VERSION_FLAT < 4004000000ULL
    params += " -mllvm --amdgpu-spill-vgpr-to-agpr=0";
#endif

#if MIOPEN_BUILD_DEV
    if(miopen::IsEnabled(MIOPEN_DEBUG_HIP_VERBOSE{}))
    {
        params += " -v";
    }
    if(miopen::IsEnabled(MIOPEN_DEBUG_HIP_DUMP{}))
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
    const std::string redirector = testing_mode ? " 1>/dev/null 2>&1" : "";
    tmp_dir->Execute(env + std::string(" ") + MIOPEN_HIP_COMPILER,
                     params + filename + " -o " + bin_file.string() + redirector);
    if(!boost::filesystem::exists(bin_file))
        MIOPEN_THROW(filename + " failed to compile");

#if defined(MIOPEN_OFFLOADBUNDLER_BIN) && !MIOPEN_BACKEND_HIP
    // Unbundling is not required for HIP runtime && hip-clang
    tmp_dir->Execute(MIOPEN_OFFLOADBUNDLER_BIN,
                     "--type=o "
#if(HIP_PACKAGE_VERSION_FLAT >= 4001021072ULL && HIP_PACKAGE_VERSION_FLAT < 4002000000ULL) || \
    HIP_PACKAGE_VERSION_FLAT >= 4002021072ULL
                     "--targets=hipv4-amdgcn-amd-amdhsa-"
#else
                     "--targets=hip-amdgcn-amd-amdhsa-"
#endif
#if HIP_PACKAGE_VERSION_FLAT < 4001000000ULL
                         + lots.device
#else
                         + (std::string{'-'} + lots.device + lots.xnack)
#endif
                         + " --inputs=" + bin_file.string() + " --outputs=" + bin_file.string() +
                         ".hsaco --unbundle");

    auto hsaco = std::find_if(boost::filesystem::directory_iterator{tmp_dir->path},
                              {},
                              [](auto entry) { return (entry.path().extension() == ".hsaco"); });

    if(hsaco == boost::filesystem::directory_iterator{})
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

#ifndef ROCM_FEATURE_LLVM_AMDGCN_BUFFER_ATOMIC_FADD_F32_RETURNS_FLOAT
static bool
HipBuildTest(const std::string& program_name, std::string params, const TargetProperties& target)
{
    boost::optional<miopen::TmpDir> dir(program_name);
    std::string source = miopen::GetKernelSrc(program_name);
    try
    {
        std::ignore = HipBuildImpl(dir, program_name, source, params, target, true);
    }
    catch(...)
    {
        return false;
    }
    return true;
}

static bool DetectIfBufferAtomicFaddReturnsFloatImpl(const TargetProperties& target)
{
    const std::string program_name("detect_llvm_amdgcn_buffer_atomic_fadd_f32_float.cpp");
    std::string params;

    if(HipBuildTest(program_name, params, target))
    {
        MIOPEN_LOG_NQI("Yes");
        return true;
    }
    MIOPEN_LOG_NQI("No");
    return false;
}

static bool DetectIfBufferAtomicFaddReturnsFloat(const TargetProperties& target)
{
    static const bool once = DetectIfBufferAtomicFaddReturnsFloatImpl(target);
    return once;
}
#endif

boost::filesystem::path HipBuild(boost::optional<TmpDir>& tmp_dir,
                                 const std::string& filename,
                                 std::string src,
                                 std::string params,
                                 const TargetProperties& target)
{
#ifndef ROCM_FEATURE_LLVM_AMDGCN_BUFFER_ATOMIC_FADD_F32_RETURNS_FLOAT
    if(miopen::solver::support_amd_buffer_atomic_fadd(target.Name()))
        if(DetectIfBufferAtomicFaddReturnsFloat(target))
            params += " -DCK_AMD_BUFFER_ATOMIC_FADD_RETURNS_FLOAT=1";
#elif ROCM_FEATURE_LLVM_AMDGCN_BUFFER_ATOMIC_FADD_F32_RETURNS_FLOAT
    if(miopen::solver::support_amd_buffer_atomic_fadd(target.Name()))
        params += " -DCK_AMD_BUFFER_ATOMIC_FADD_RETURNS_FLOAT=1";
#endif
    return HipBuildImpl(tmp_dir, filename, src, params, target, false);
}

void bin_file_to_str(const boost::filesystem::path& file, std::string& buf)
{
    std::ifstream bin_file_ptr(file.string().c_str(), std::ios::binary);
    std::ostringstream bin_file_strm;
    bin_file_strm << bin_file_ptr.rdbuf();
    buf = bin_file_strm.str();
}

} // namespace miopen
