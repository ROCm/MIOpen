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

#include "test.hpp"
#include "driver.hpp"
#include <miopen/kernel_build_params.hpp>

namespace miopen {
namespace tests {
struct KBPTestDriver : test_driver
{
    void run() const
    {
        auto kbp = KernelBuildParameters{{"TrivialDefine"},
                                         {"DefineWithValue", 0},
                                         {kbp::Option{}, "TrivialOption"},
                                         {kbp::Option{}, "OptionWithValue", 0}}
                   << KernelBuildParameters{{"Shifted"}};
        kbp.Define("DefineDefine");
        kbp.Define("DefineDefineWithValue", 1);
        EXPECT_EQUAL(kbp.GenerateFor(kbp::OpenCL{}),
                     "-DTrivialDefine -DDefineWithValue=0 -TrivialOption -OptionWithValue 0 "
                     "-DShifted -DDefineDefine "
                     "-DDefineDefineWithValue=1");
        EXPECT_EQUAL(
            kbp.GenerateFor(kbp::GcnAsm{}),
            "-Wa,-defsym,TrivialDefine "
            "-Wa,-defsym,DefineWithValue=0 -TrivialOption -OptionWithValue 0 -Wa,-defsym,Shifted "
            "-Wa,-defsym,DefineDefine "
            "-Wa,-defsym,DefineDefineWithValue=1");
    }
};
} // namespace tests
} // namespace miopen

int main(int argc, const char** argn) { test_drive<miopen::tests::KBPTestDriver>(argc, argn); }
