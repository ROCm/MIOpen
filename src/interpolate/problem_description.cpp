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

#include <miopen/interpolate/problem_description.hpp>
#include <miopen/names.hpp>

#include <vector>
#include <sstream>

namespace miopen {

namespace interpolate {

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
    auto input_dims              = inputDesc.GetLengths();
    auto output_dims             = outputDesc.GetLengths();
    auto input_dtype             = inputDesc.GetType();
    auto Si                      = inputDesc.GetStrides();
    auto So                      = outputDesc.GetStrides();
    miopenInterpolateMode_t mode = GetMode();
    bool align_corners           = GetAlignCorners();

    std::ostringstream ss;
    ss << "interpolate";
    ss << "is_fwd" << is_fwd;
    ss << "mode" << mode;
    ss << "align_corners" << align_corners;
    ss << "input_dtype" << input_dtype;
    ss << "input_dims" << input_dims;
    ss << "input_stride" << Si;
    ss << "output_dims" << output_dims;
    ss << "output_stride" << So;

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto input_grad_dims         = inputGradDesc.GetLengths();
    auto output_grad_dims        = outputGradDesc.GetLengths();
    auto output_dtype            = outputGradDesc.GetType();
    auto Si                      = inputGradDesc.GetStrides();
    auto So                      = outputGradDesc.GetStrides();
    miopenInterpolateMode_t mode = GetMode();
    bool align_corners           = GetAlignCorners();

    std::ostringstream ss;
    ss << "interpolate";
    ss << "is_fwd" << is_fwd;
    ss << "mode" << mode;
    ss << "align_corners" << align_corners;
    ss << "output_grad_dtype" << output_dtype;
    ss << "output_grad_dims" << output_grad_dims;
    ss << "output_grad_stride" << So;
    ss << "input_grad_dims" << input_grad_dims;
    ss << "input_grad_stride" << Si;

    return NetworkConfig{ss.str()};
}

} // namespace interpolate

} // namespace miopen
