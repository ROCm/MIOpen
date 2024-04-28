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
        for(const auto& d : desc.GetStrides())
        {
            ss << d << "x";
        }
    };

    if(isForward)
    {
        ss << "fwd-";
        for(auto s : mhaInputDescsForwardPtr->oDesc.GetLengths())
        {
            ss << s << "x";
        }

        print_strides(mhaInputDescsForwardPtr->kDesc);
        print_strides(mhaInputDescsForwardPtr->qDesc);
        print_strides(mhaInputDescsForwardPtr->vDesc);
        print_strides(mhaInputDescsForwardPtr->oDesc);
        print_strides(mhaInputDescsForwardPtr->mDesc);
        print_strides(mhaInputDescsForwardPtr->zInvDesc);

        ss << mhaInputDescsForwardPtr->oDesc.GetType();
    }
    else
    {
        ss << "bwd-";

        for(auto s : mhaInputDescsBackwardPtr->oDesc.GetLengths())
        {
            ss << s << "x";
        }
        print_strides(mhaInputDescsBackwardPtr->kDesc);
        print_strides(mhaInputDescsBackwardPtr->qDesc);
        print_strides(mhaInputDescsBackwardPtr->vDesc);
        print_strides(mhaInputDescsBackwardPtr->oDesc);
        print_strides(mhaInputDescsBackwardPtr->doDesc);
        print_strides(mhaInputDescsBackwardPtr->mDesc);
        print_strides(mhaInputDescsBackwardPtr->zInvDesc);
        print_strides(mhaInputDescsBackwardPtr->dqDesc);
        print_strides(mhaInputDescsBackwardPtr->dkDesc);
        print_strides(mhaInputDescsBackwardPtr->dvDesc);

        ss << mhaInputDescsBackwardPtr->oDesc.GetType();
    }

    return NetworkConfig{ss.str()};
}

} // namespace mha

} // namespace miopen
