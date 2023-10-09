/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/pooling/problem_description.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/pooling.hpp>

#include <sstream>

namespace miopen {

namespace pooling {

namespace {

template <typename T>
std::string get_vect_config(const std::vector<T>& v)
{
    std::string str;
    for(auto itr = v.begin(); itr < v.end(); itr++)
    {
        str += (std::to_string(*itr) + (itr == v.end() - 1 ? "" : "x"));
    }
    return str;
}

} // namespace

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    int pooling_method =
        (pooling.GetMode() == miopenPoolingMax)
            ? MLO_POOLING_OP_MAX
            : ((pooling.GetMode() == miopenPoolingAverage) ? MLO_POOLING_OP_AVE
                                                           : MLO_POOLING_OP_AVE_INCLUSIVE);

    ss << "m" + std::to_string(pooling_method);
    ss << "_dt" << xDesc.GetType();
    if(const auto ct = xDesc.GetCastType())
        ss << "_dct" << GetDataTypeName(*ct);
    ss << "_ker" << get_vect_config(pooling.lens);
    ss << "_str" << get_vect_config(pooling.strides);
    ss << "_pad" << get_vect_config(pooling.pads);
    ss << "_it" << pooling.GetIndexType();
    ss << "_im" << pooling.GetWorkspaceIndexMode();
    if(direction == Direction::Forward)
    {
        ss << "_is" << static_cast<int>(save_index);
    }
    ss << "_xd" << get_vect_config(xDesc.GetLengths());
    ss << "_xs" << get_vect_config(xDesc.GetStrides());
    ss << "_yd" << get_vect_config(yDesc.GetLengths());
    ss << "_ys" << get_vect_config(yDesc.GetStrides());
    if(direction == Direction::Backward)
    {
        ss << "_dxd" << get_vect_config(dxDesc.GetLengths());
        ss << "_dxs" << get_vect_config(dxDesc.GetStrides());
        ss << "_dyd" << get_vect_config(dyDesc.GetLengths());
        ss << "_dys" << get_vect_config(dyDesc.GetStrides());
    }

    return NetworkConfig{ss.str()};
}

} // namespace pooling

} // namespace miopen
