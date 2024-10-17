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

#include "miopen/names.hpp"
#include <miopen/mseloss/problem_description.hpp>
#include <sstream>

namespace miopen {
namespace mseloss {
namespace forward {
NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dtype = xDesc.GetType();

    std::ostringstream ss;
    ss << "fwd";
    ss << "dtype" << dtype;
    ss << "xDesc" << xDesc;
    ss << "yDesc" << yDesc;

    return NetworkConfig{ss.str()};
}

} // namespace forward

namespace forward_unreduced {
NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dtype = xDesc.GetType();

    std::ostringstream ss;
    ss << "fwdu";
    ss << "dtype" << dtype;
    ss << "xdesc" << xDesc;
    ss << "yDesc" << yDesc;

    return NetworkConfig{ss.str()};
}
} // namespace forward_unreduced

namespace backward {
NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dtype = xDesc.GetType();

    std::ostringstream ss;
    ss << "bwd";
    ss << "dtype" << dtype;
    ss << "dxDesc" << dxDesc;
    ss << "dyDesc" << dyDesc;
    ss << "zDesc" << zDesc;

    return NetworkConfig{ss.str()};
}

} // namespace backward

namespace backward_unreduced {
NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dtype = xDesc.GetType();

    std::ostringstream ss;
    ss << "bwdu";
    ss << "dtype" << dtype;
    ss << "dxDesc" << dxDesc;
    ss << "dyDesc" << dyDesc;
    ss << "zDesc" << zDesc;

    return NetworkConfig{ss.str()};
}

} // namespace backward_unreduced
} // namespace mseloss
} // namespace miopen
