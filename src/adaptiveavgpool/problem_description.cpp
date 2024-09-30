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

#include <miopen/adaptiveavgpool/problem_description.hpp>
#include <miopen/names.hpp>
#include <sstream>

namespace miopen {

namespace adaptiveavgpool {

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

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    auto input_size    = inputDesc.GetLengths();
    auto output_size   = outputDesc.GetLengths();
    auto input_stride  = inputDesc.GetStrides();
    auto output_stride = outputDesc.GetStrides();

    auto input_dtype = inputDesc.GetType();

    std::ostringstream ss;

    ss << "adaptiveavgpool_fwd";
    ss << "-input_dtype" << input_dtype;
    ss << "-Is" << input_size;
    ss << "-Os" << output_size;
    ss << "-Si" << input_stride;
    ss << "-So" << output_stride;

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto input_grad_size    = inputGradDesc.GetLengths();
    auto output_grad_size   = outputGradDesc.GetLengths();
    auto input_grad_stride  = inputGradDesc.GetStrides();
    auto output_grad_stride = outputGradDesc.GetStrides();

    auto input_dtype = inputGradDesc.GetType();

    std::ostringstream ss;

    ss << "adaptiveavgpool_bwd";
    ss << "-input_dtype" << input_dtype;
    ss << "-dIs" << input_grad_size;
    ss << "-dOs" << output_grad_size;
    ss << "-dSi" << input_grad_stride;
    ss << "-dSo" << output_grad_stride;

    return NetworkConfig{ss.str()};
}

} // namespace adaptiveavgpool

} // namespace miopen
