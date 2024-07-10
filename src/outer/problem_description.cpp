/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/outer/problem_description.hpp>
#include <miopen/datatype.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace outer {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    if(is_fwd == true)
    {
        ss << "outerfwd";
    }
    else
    {
        if(grad == ONE)
        {
            ss << "outerbwdgrad1";
        }
        else if(grad == TWO)
        {
            ss << "outerbwdgrad2";
        }
    }
    auto x1length = x1Desc.GetLengths();
    auto x2length = x2Desc.GetLengths();
    auto ylength  = yDesc.GetLengths();
    auto dtype    = x1Desc.GetType();
    ss << "dtype" << dtype;
    ss << "x1len" << x1length[0];
    ss << "x2len" << x2length[0];

    return NetworkConfig{ss.str()};
}

} // namespace outer

} // namespace miopen
