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
#include <sstream>

#include <miopen/median/problem_description.hpp>

namespace miopen {

namespace median {

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype         = inputDesc.GetType();
    auto input_lengths = inputDesc.GetLengths();

    std::ostringstream ss;
    ss << "median_fwd";
    ss << "dtype" << dtype;
    ss << "input_lengths";
    for(auto length : input_lengths)
        ss << length << ',';
    ss << "dim_stride" << inputDesc.GetStrides()[dim];

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype              = inputGradDesc.GetType();
    auto input_grad_lengths = inputGradDesc.GetLengths();

    std::ostringstream ss;
    ss << "median_bwd";
    ss << "dtype" << dtype;
    ss << "input_grad_lengths";
    for(auto length : input_grad_lengths)
        ss << length << ',';
    ss << "dim_stride" << inputGradDesc.GetStrides()[dim];

    return NetworkConfig{ss.str()};
}

} // namespace median

} // namespace miopen
