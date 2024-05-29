/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/datatype.hpp>
#include <miopen/softmax/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>
#include <string_view>

namespace miopen {

namespace softmax {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss(isForward ? "sfmfwd-" : "sfmbwd-");

    // all the tensors must be the same size and types
    // so we can use only one set of values
    const auto& desc            = isForward ? xdxDesc : yDesc;
    const auto [sn, sc, sh, sw] = tien<4>(desc.GetLengths());
    ss << "n" << sn << "c" << sc << "h" << sh << "w" << sw;
    ss << GetDataType(desc.GetType());
    ss << "a" << alpha;
    ss << "b" << beta;
    ss << "algo" << static_cast<int>(algorithm);
    ss << "mode" << static_cast<int>(mode);

    auto printStrides = [&ss](std::string_view name, const miopen::TensorDescriptor& d) {
        if(d.IsPacked())
        {
            ss << name << "pk1";
        }
        else
        {
            const auto [n, c, h, w] = tien<4>(d.GetStrides());
            ss << name << "pk0strides" << n << "x" << c << "x" << h << "x" << w;
        }
    };

    if(isForward)
    {
        printStrides("x", xdxDesc);
        printStrides("y", yDesc);
    }
    else
    {
        printStrides("y", yDesc);
        printStrides("dy", dyDesc);
        printStrides("dx", xdxDesc);
    }

    return NetworkConfig{ss.str()};
}

} // namespace softmax

} // namespace miopen
