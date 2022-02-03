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

#ifdef __linux__
#include <paths.h>
#else
#include <Windows.h>
#endif // __linux__

#include <string>
#include <fstream>

#include <boost/filesystem.hpp>

#include <miopen/errors.hpp>
#include <miopen/tmp_dir.hpp>

#include "test.hpp"

namespace bf = boost::filesystem;

static int Child(const std::string& path, std::string cmd)
{
#ifdef __linux__
    std::ignore = path;
    auto pipe   = popen(cmd.c_str(), "w");

    if(pipe == nullptr)
        MIOPEN_THROW("Error: popen()");

    auto status = pclose(pipe);
    return WEXITSTATUS(status);
#else
    STARTUPINFOA info{sizeof(info)};
    PROCESS_INFORMATION processInfo{};

    if(!CreateProcessA(
           path, &cmd[0], nullptr, nullptr, false, 0, nullptr, nullptr, &info, &processInfo))
        MIOPEN_THROW("CreateProcess error: " << GetLastError());

    WaitForSingleObject(processInfo.hProcess, INFINITE);

    DWORD status;
    const auto getExitCodeStatus = GetExitCodeProcess(processInfo.hProcess, &status);

    CloseHandle(processInfo.hProcess);
    CloseHandle(processInfo.hThread);

    if(!getExitCodeStatus)
        MIOPEN_THROW("GetExitCodeProcess error: " << GetLastError());

    return status;
#endif // __linux__
}

namespace miopen {
namespace tests {

class InlinerTest
{
public:
    void Run(const bf::path& exe_path) const
    {
        const TmpDir test_srcs{"test_include_inliner"};
        const auto addkernels      = (exe_path.parent_path() / "addkernels").string();
        const auto header_filename = "header.h";
        const auto asm_src         = test_srcs.path / "valid.s";
        const auto valid_src       = test_srcs.path / "valid.cl";
        const auto invalid_src     = test_srcs.path / "invalid.cl";
        const auto header_src      = test_srcs.path / header_filename;

        // clang-format-off
        std::ofstream(valid_src.c_str()) << "#include <" << header_filename << ">" << std::endl
                                         << "#include \"" << header_filename << "\"" << std::endl
                                         << "//inliner-include-optional" << std::endl
                                         << "#include <missing_header.h>" << std::endl;
        // clang-format-on

        std::ofstream(asm_src.c_str()) << ".include \"" << header_filename << "\"" << std::endl;
        std::ofstream(invalid_src.c_str()) << "#include <missing_header.h>" << std::endl;
        std::ofstream(header_src.c_str()) << std::endl;

        EXPECT_EQUAL(0, Child(addkernels, addkernels + " -source " + valid_src.string()));
        EXPECT_EQUAL(0, Child(addkernels, addkernels + " -source " + asm_src.string()));
        EXPECT_EQUAL(1, Child(addkernels, addkernels + " -source " + invalid_src.string()));
    }
};

} // namespace tests
} // namespace miopen

int main(int, const char** cargs)
{
    miopen::tests::InlinerTest{}.Run(cargs[0]);
    return 0;
}
