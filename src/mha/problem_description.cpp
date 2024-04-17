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

#include <miopen/mha/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace mha {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    ss << "mha";

    auto print_strides = [&ss](const TensorDescriptor& desc) {
        for(auto d : desc.GetStrides())
        {
            ss << d << "x";
        }
    };

    if(isForward)
    {
        ss << "fwd-";
        for(auto s : mhaInputDescsForward.oDesc.GetLengths())
        {
            ss << s << "x";
        }

        print_strides(mhaInputDescsForward.kDesc);
        print_strides(mhaInputDescsForward.qDesc);
        print_strides(mhaInputDescsForward.vDesc);
        print_strides(mhaInputDescsForward.oDesc);
        print_strides(mhaInputDescsForward.mDesc);
        print_strides(mhaInputDescsForward.zInvDesc);

        ss << mhaInputDescsForward.oDesc.GetType();
    }
    else
    {
        ss << "bwd-";

        for(auto s : mhaInputDescsBackward.oDesc.GetLengths())
        {
            ss << s << "x";
        }
        print_strides(mhaInputDescsBackward.kDesc);
        print_strides(mhaInputDescsBackward.qDesc);
        print_strides(mhaInputDescsBackward.vDesc);
        print_strides(mhaInputDescsBackward.oDesc);
        print_strides(mhaInputDescsBackward.doDesc);
        print_strides(mhaInputDescsBackward.mDesc);
        print_strides(mhaInputDescsBackward.zInvDesc);
        print_strides(mhaInputDescsBackward.dqDesc);
        print_strides(mhaInputDescsBackward.dkDesc);
        print_strides(mhaInputDescsBackward.dvDesc);

        ss << mhaInputDescsBackward.oDesc.GetType();
    }

    return NetworkConfig{ss.str()};
}

} // namespace mha

} // namespace miopen
