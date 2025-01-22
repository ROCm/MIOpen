/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <miopen/fractionalmaxpool/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace fractionalmaxpool {

inline std::ostream& operator<<(std::ostream& os, const std::vector<uint64_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype         = outputDesc.GetType();
    auto indices_dtype = indicesDesc.GetType();
    std::ostringstream ss;

    ss << "fractionalmaxpool_fwd";
    ss << "-dtype" << dtype;
    ss << "-indices_dtype" << indices_dtype;
    ss << "-Is" << inputDesc.GetLengths();
    ss << "-Os" << outputDesc.GetLengths();

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype         = outputGradDesc.GetType();
    auto indices_dtype = indicesDesc.GetType();
    std::ostringstream ss;

    ss << "fractionalmaxpool_bwd";
    ss << "-dtype" << dtype;
    ss << "-indices_dtype" << indices_dtype;
    ss << "-dIs" << inputGradDesc.GetLengths();
    ss << "-dOs" << outputGradDesc.GetLengths();

    return NetworkConfig{ss.str()};
}

} // namespace fractionalmaxpool

} // namespace miopen
