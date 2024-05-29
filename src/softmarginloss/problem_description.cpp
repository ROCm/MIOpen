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
#include <miopen/softmarginloss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace softmarginloss {

NetworkConfig ForwardProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    ss << "softmarginloss_fwd";
    ss << "itype" << iDesc.GetType();
    ss << "ttype" << tDesc.GetType();
    ss << "otype" << oDesc.GetType();
    ss << "ilen";
    auto ilen = iDesc.GetLengths();
    for(int32_t i = 0; i < ilen.size(); i++)
        ss << ilen[i] << "_";
    ss << "tlen";
    auto tlen = tDesc.GetLengths();
    for(int32_t i = 0; i < tlen.size(); i++)
        ss << tlen[i] << "_";
    ss << "olen";
    auto olen = oDesc.GetLengths();
    for(int32_t i = 0; i < olen.size(); i++)
        ss << olen[i] << "_";
    return NetworkConfig{ss.str()};
}

NetworkConfig BackwardProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    ss << "softmarginloss_bwd";
    ss << "itype" << iDesc.GetType();
    ss << "ttype" << tDesc.GetType();
    ss << "dOtype" << dODesc.GetType();
    ss << "dItype" << dIDesc.GetType();
    ss << "ilen";
    auto ilen = iDesc.GetLengths();
    for(int32_t i = 0; i < ilen.size(); i++)
        ss << ilen[i] << "_";
    ss << "tlen";
    auto tlen = tDesc.GetLengths();
    for(int32_t i = 0; i < tlen.size(); i++)
        ss << tlen[i] << "_";
    ss << "dOlen";
    auto dOlen = dODesc.GetLengths();
    for(int32_t i = 0; i < dOlen.size(); i++)
        ss << dOlen[i] << "_";
    ss << "dIlen";
    auto dIlen = dIDesc.GetLengths();
    for(int32_t i = 0; i < dIlen.size(); i++)
        ss << dIlen[i] << "_";
    return NetworkConfig{ss.str()};
}

} // namespace softmarginloss

} // namespace miopen
