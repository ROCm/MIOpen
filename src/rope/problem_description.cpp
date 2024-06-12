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

#include <miopen/rope/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace rope {

NetworkConfig ProblemDescriptionFwd::MakeNetworkConfig() const
{
    std::ostringstream ss;
    ss << "rope";

    auto print_strides = [&ss](const TensorDescriptor& desc) {
        for(const auto& d : desc.GetStrides())
        {
            ss << d << "x";
        }
    };

    auto ylength = yDesc.GetLengths();

    auto output_numel = std::accumulate(
        ylength.begin(), ylength.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    auto dtype = xDesc.GetType();

    print_strides(xDesc);
    print_strides(cosDesc);
    print_strides(sinDesc);
    print_strides(yDesc);

    ss << "fwd-";
    ss << "dtype" << dtype;
    ss << "output_numel" << output_numel;

    return NetworkConfig{ss.str()};
}

NetworkConfig ProblemDescriptionBwd::MakeNetworkConfig() const
{
    std::ostringstream ss;
    ss << "rope";

    auto print_strides = [&ss](const TensorDescriptor& desc) {
        for(const auto& d : desc.GetStrides())
        {
            ss << d << "x";
        }
    };

    auto dxlength = dxDesc.GetLengths();

    auto output_numel = std::accumulate(
        dxlength.begin(), dxlength.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    auto dtype = dyDesc.GetType();

    print_strides(dyDesc);
    print_strides(cosDesc);
    print_strides(sinDesc);
    print_strides(dxDesc);

    ss << "bwd-";
    ss << "dtype" << dtype;
    ss << "output_numel" << output_numel;

    return NetworkConfig{ss.str()};
}

} // namespace rope

} // namespace miopen
