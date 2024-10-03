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

#include <cstddef>
#include <miopen/nllloss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace nllloss {

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
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

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto input_size     = inputDesc.GetLengths();
    auto input_strides  = inputDesc.GetStrides();
    auto target_size    = targetDesc.GetLengths();
    auto target_strides = targetDesc.GetStrides();
    auto weight_size    = weightDesc.GetLengths();
    auto weight_strides = weightDesc.GetStrides();
    auto output_size    = outputDesc.GetLengths();
    auto output_strides = outputDesc.GetStrides();

    auto input_dtype = inputDesc.GetType();

    std::ostringstream ss;

    ss << "nllloss";
    ss << "is_fwd" << is_fwd;
    ss << "reduction" << reduction;
    ss << "-input_dtype" << input_dtype;
    ss << "-dIs" << input_size;
    ss << "-dOs" << output_size;
    ss << "-dSi" << input_strides;
    ss << "-dSo" << output_strides;
    ss << "-dTs" << target_size;
    ss << "-dTs" << target_strides;
    ss << "-dWs" << weight_size;
    ss << "-dWs" << weight_strides;

    return NetworkConfig{ss.str()};
}

} // namespace nllloss

} // namespace miopen
