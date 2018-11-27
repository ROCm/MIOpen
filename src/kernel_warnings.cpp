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
#include <iterator>
#include <miopen/config.h>
#include <miopen/kernel_warnings.hpp>
#include <miopen/stringutils.hpp>
#include <numeric>
#include <sstream>

namespace miopen {

std::vector<std::string> KernelWarnings()
{
    return {
        "-Weverything",
        "-Wno-shorten-64-to-32",
        "-Wno-unused-macros",
        "-Wno-unused-function",
        "-Wno-sign-compare",
        "-Wno-reserved-id-macro",
        "-Wno-sign-conversion",
        "-Wno-missing-prototypes",
        "-Wno-cast-qual",
        "-Wno-cast-align",
        "-Wno-conversion",
        "-Wno-double-promotion",
        "-Wno-float-equal",
    };
}

std::string MakeKernelWarningsString()
{
#if MIOPEN_BACKEND_OPENCL
    std::string prefix = " -Wf,";

#else
    std::string prefix = " ";
#endif
    return prefix + JoinStrings(KernelWarnings(), prefix);
}

const std::string& KernelWarningsString()
{
    static const std::string result = MakeKernelWarningsString();
    return result;
}

} // namespace miopen
