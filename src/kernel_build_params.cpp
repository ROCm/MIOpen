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

#include <sstream>

#include <boost/range/adaptor/transformed.hpp>

#include <miopen/kernel_build_params.hpp>
#include <miopen/stringutils.hpp>

namespace miopen {

static std::string GenerateDefines(const std::vector<KernelBuildParameter>& options,
                                   const std::string& prefix)
{
    const auto strs =
        options | boost::adaptors::transformed([&prefix](const KernelBuildParameter& define) {
            std::ostringstream ss;

            ss << '-';
            if(define.type == ParameterTypes::Define)
                ss << prefix;

            ss << define.name;

            if(!define.value.empty())
            {
                switch(define.type)
                {
                case ParameterTypes::Define: ss << '='; break;
                case ParameterTypes::Option: ss << ' '; break;
                }

                ss << define.value;
            }

            return ss.str();
        });

    return JoinStrings(strs, " ");
}

std::string kbp::OpenCL::Generate(const std::vector<KernelBuildParameter>& options)
{
    // Ensure only one space after the -cl-std.
    // >1 space can cause an Apple compiler bug. See clSPARSE issue #141.

    return GenerateDefines(options, "D");
}

std::string kbp::GcnAsm::Generate(const std::vector<KernelBuildParameter>& options)
{
    return GenerateDefines(options, "Wa,-defsym,");
}

} // namespace miopen
