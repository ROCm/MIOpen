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

#include <miopen/sparse_softmax_cross_entropy_with_logits/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace sparse_softmax_cross_entropy_with_logits {

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
    auto dtype = outputDesc.GetType();
    std::ostringstream ss;

    ss << "sparse_softmax_cross_entropy_with_logits_fwd";
    ss << "-dtype" << dtype;
    ss << "-Is" << inputDesc.GetLengths();
    ss << "-IsContiguous" << IsAllContiguous();

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype = outputGradDesc.GetType();
    std::ostringstream ss;

    ss << "sparse_softmax_cross_entropy_with_logits_bwd";
    ss << "-dtype" << dtype;
    ss << "-dIs" << inputGradDesc.GetLengths();
    ss << "-IsContiguous" << IsAllContiguous();

    return NetworkConfig{ss.str()};
}

} // namespace sparse_softmax_cross_entropy_with_logits

} // namespace miopen
