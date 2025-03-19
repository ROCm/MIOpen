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

#include <miopen/names.hpp>
#include <miopen/pad_constant/problem_description.hpp>

#include <sstream>

namespace miopen {
namespace pad_constant {

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype = xDesc.GetType();

    std::ostringstream ss;
    ss << "padconstant_fwd";
    ss << "dtype" << dtype;
    ss << "contiguous" << IsContiguous();
    ss << "input_dims" << xDesc.GetNumDims();
    ss << "output_size" << yDesc.GetElementSize();
    auto padding_vec = GetPadding();
    ss << "padding_values";
    for(auto i : padding_vec)
        ss << i << ",";

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype = dxDesc.GetType();

    std::ostringstream ss;
    ss << "padconstant_bwd";
    ss << "dtype" << dtype;
    ss << "contiguous" << IsContiguous();
    ss << "input_dims" << dxDesc.GetNumDims();
    ss << "output_size" << dyDesc.GetElementSize();
    ss << "padding_values";
    auto padding_vec = GetPadding();
    for(auto i : padding_vec)
        ss << i << ",";

    return NetworkConfig{ss.str()};
}
} // namespace pad_constant
} // namespace miopen
