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

#include <miopen/pad_reflection/problem_description.hpp>

#include <sstream>

namespace miopen {
namespace pad_reflection {

NetworkConfig PadReflectionFwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype = xDesc.GetType();

    auto output_numel = yDesc.GetElementSize();

    std::ostringstream ss;
    ss << "pad_reflection_fwd";
    ss << "dtype" << dtype;
    ss << "output_numel" << output_numel;
    ss << "is_contiguous" << IsContiguous();

    return NetworkConfig{ss.str()};
}

NetworkConfig PadReflectionBwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype          = dxDesc.GetType();
    auto output_numel   = dyDesc.GetElementSize();
    auto input_last_dim = dxDesc.GetLengths().back();

    std::ostringstream ss;
    ss << "pad_reflection_bwd";
    ss << "dtype" << dtype;
    ss << "output_numel" << output_numel;
    ss << "input_last_dim" << input_last_dim;

    return NetworkConfig{ss.str()};
}

} // namespace pad_reflection
} // namespace miopen
