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

#include <string>
#include <fstream>

#include <miopen/filesystem.hpp>
#include <miopen/errors.hpp>
#include <miopen/tmp_dir.hpp>
#include <miopen/process.hpp>

#include "test.hpp"

namespace miopen {
namespace tests {

static int Child(std::string_view cmd, const fs::path& path)
{
    return miopen::Process{cmd}("-source " + path);
}

class InlinerTest
{
public:
    void Run(const fs::path& exe_path) const
    {
        const TmpDir test_srcs{"test_include_inliner"};
        const auto addkernels =
            miopen::make_executable_name(exe_path.parent_path() / "addkernels").string();
        const auto header_filename = "header.h";
        const auto asm_src         = test_srcs.path / "valid.s";
        const auto valid_src       = test_srcs.path / "valid.cl";
        const auto invalid_src     = test_srcs.path / "invalid.cl";
        const auto header_src      = test_srcs.path / header_filename;

        // clang-format-off
        std::ofstream(valid_src.c_str()) << "#include <" << header_filename << ">\n"
                                         << "#include \"" << header_filename << "\"\n"
                                         << "//inliner-include-optional\n"
                                         << "#include <missing_header.h>" << std::endl;
        // clang-format-on

        std::ofstream(asm_src.c_str()) << ".include \"" << header_filename << "\"" << std::endl;
        std::ofstream(invalid_src.c_str()) << "#include <missing_header.h>" << std::endl;
        std::ofstream(header_src.c_str()) << std::endl;

        EXPECT_EQUAL(0, Child(addkernels, valid_src));
        EXPECT_EQUAL(0, Child(addkernels, asm_src));
        EXPECT_EQUAL(1, Child(addkernels, invalid_src));
    }
};

} // namespace tests
} // namespace miopen

int main(int, const char** cargs)
{
    miopen::tests::InlinerTest{}.Run(cargs[0]);
    return 0;
}
