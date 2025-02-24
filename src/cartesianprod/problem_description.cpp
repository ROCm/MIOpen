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

#include <miopen/errors.hpp>
#include <miopen/cartesianprod/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace cartesianprod {

inline std::ostream& operator<<(std::ostream& os, const std::vector<uint64_t>& v)
{
    os << '{';
    for(size_t i = 0; i < v.size(); ++i)
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
    auto output_size = outputDesc.GetLengths();

    auto dtype = outputDesc.GetType();

    std::ostringstream ss;

    ss << "cartesianprod_fwd";
    ss << "-dtype" << dtype;
    ss << "-Is";
    for(size_t i = 0; i < inputCount; i++)
    {
        ss << "_" << deref(inputDescs[i]).GetLengths();
    }
    ss << "-Os" << output_size;
    ss << "-Ic" << IsAllPacked();

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto output_grad_size = outputGradDesc.GetLengths();

    auto dtype = outputGradDesc.GetType();

    std::ostringstream ss;

    ss << "cartesianprod_bwd";
    ss << "-dtype" << dtype;
    ss << "-dIs";
    for(size_t i = 0; i < inputCount; i++)
    {
        ss << "_" << deref(inputGradDescs[i]).GetLengths();
    }
    ss << "-dOs" << output_grad_size;
    ss << "-Ic" << IsAllPacked();

    return NetworkConfig{ss.str()};
}

} // namespace cartesianprod

} // namespace miopen
