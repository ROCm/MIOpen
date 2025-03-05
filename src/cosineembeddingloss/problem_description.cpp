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

#include <miopen/cosineembeddingloss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace cosineembeddingloss {

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
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

NetworkConfig FwdUnreducedProblemDescription::MakeNetworkConfig() const
{
    auto input_dims  = input1Desc.GetLengths();
    auto input_dtype = input1Desc.GetType();
    auto Si1         = input1Desc.GetStrides();
    auto Si2         = input2Desc.GetStrides();
    auto St          = targetDesc.GetStrides();
    auto So          = outputDesc.GetStrides();

    std::ostringstream ss;
    ss << "cosineembeddingloss_unreduced";
    ss << "is_fwd" << is_fwd;
    ss << "input_dtype" << input_dtype;
    ss << "input_dims" << input_dims;
    ss << "input1_stride" << Si1;
    ss << "input2_stride" << Si2;
    ss << "target_stride" << St;
    ss << "output_stride" << So;

    return NetworkConfig{ss.str()};
}

NetworkConfig FwdReducedProblemDescription::MakeNetworkConfig() const
{
    auto input_dims  = input1Desc.GetLengths();
    auto input_dtype = input1Desc.GetType();
    auto Si1         = input1Desc.GetStrides();
    auto Si2         = input2Desc.GetStrides();
    auto St          = targetDesc.GetStrides();
    auto So          = outputDesc.GetStrides();

    std::ostringstream ss;
    ss << "cosineembeddingloss_reduced";
    ss << "is_fwd" << is_fwd;
    ss << "input_dtype" << input_dtype;
    ss << "input_dims" << input_dims;
    ss << "input1_stride" << Si1;
    ss << "input2_stride" << Si2;
    ss << "target_stride" << St;
    ss << "output_stride" << So;

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdUnreducedProblemDescription::MakeNetworkConfig() const
{
    auto input_dims  = input1Desc.GetLengths();
    auto input_dtype = input1Desc.GetType();
    auto Si1         = input1Desc.GetStrides();
    auto Si2         = input2Desc.GetStrides();
    auto St          = targetDesc.GetStrides();
    auto So          = outputDesc.GetStrides();
    auto Sg1         = input1GradDesc.GetStrides();
    auto Sg2         = input2GradDesc.GetStrides();

    std::ostringstream ss;
    ss << "cosineembeddingloss_unreduce";
    ss << "is_fwd" << is_fwd;
    ss << "input_dtype" << input_dtype;
    ss << "input_dims" << input_dims;
    ss << "input1_stride" << Si1;
    ss << "input2_stride" << Si2;
    ss << "target_stride" << St;
    ss << "output_grad_stride" << So;
    ss << "input1_grad_stride" << Sg1;
    ss << "input2_grad_stride" << Sg2;

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdReducedProblemDescription::MakeNetworkConfig() const
{
    auto input_dims  = input1Desc.GetLengths();
    auto input_dtype = input1Desc.GetType();
    auto Si1         = input1Desc.GetStrides();
    auto Si2         = input2Desc.GetStrides();
    auto St          = targetDesc.GetStrides();
    auto So          = outputDesc.GetStrides();
    auto Sg1         = input1GradDesc.GetStrides();
    auto Sg2         = input2GradDesc.GetStrides();

    std::ostringstream ss;
    ss << "cosineembeddingloss_reduce";
    ss << "is_fwd" << is_fwd;
    ss << "input_dtype" << input_dtype;
    ss << "input_dims" << input_dims;
    ss << "input1_stride" << Si1;
    ss << "input2_stride" << Si2;
    ss << "target_stride" << St;
    ss << "output_grad_stride" << So;
    ss << "input1_grad_stride" << Sg1;
    ss << "input2_grad_stride" << Sg2;

    return NetworkConfig{ss.str()};
}

} // namespace cosineembeddingloss

} // namespace miopen
