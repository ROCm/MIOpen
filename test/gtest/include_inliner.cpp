/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <fstream>
#include <miopen/tmp_dir.hpp>

#include <gtest/gtest.h>

TEST(CPU_kernel_inliner_NONE, InlinerTest)
{
    const miopen::TmpDir test_srcs{"test_include_inliner"};

    const auto bin_path   = miopen::fs::path(::testing::internal::GetArgvs().front()).parent_path();
    const auto addkernels = miopen::make_executable_name(bin_path / "addkernels").string();

    auto Child = [&](const miopen::fs::path& source) {
        return test_srcs.Execute(addkernels, "-source " + source);
    };

    const auto header_filename = "header.h";
    const auto asm_src         = test_srcs / "valid.s";
    const auto valid_src       = test_srcs / "valid.cl";
    const auto invalid_src     = test_srcs / "invalid.cl";
    const auto header_src      = test_srcs / header_filename;

    std::ofstream(valid_src.c_str()) << "#include <" << header_filename << ">\n"    //
                                     << "#include \"" << header_filename << "\"\n"  //
                                     << "//inliner-include-optional\n"              //
                                     << "#include <missing_header.h>" << std::endl; //

    std::ofstream(asm_src.c_str()) << ".include \"" << header_filename << "\"" << std::endl;
    std::ofstream(invalid_src.c_str()) << "#include <missing_header.h>" << std::endl;
    std::ofstream(header_src.c_str()) << std::endl;

    EXPECT_EQ(0, Child(valid_src));
    EXPECT_EQ(0, Child(asm_src));
    EXPECT_EQ(1, Child(invalid_src));
}
