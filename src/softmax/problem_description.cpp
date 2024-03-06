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

#include <miopen/softmax/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace softmax {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;

    ss << "sfmfwd-";

    if(isForward)
    {
        int n_x, c_x, h_x, w_x;
        int n_y, c_y, h_y, w_y;

        std::tie(n_x, c_x, h_x, w_x) = tien<4>(xdxDesc.GetLengths());
        std::tie(n_y, c_y, h_y, w_y) = tien<4>(yDesc.GetLengths());

        ss << "n_x" << n_x << "c_x" << c_x << "h_x" << h_x << "w_x" << w_x;
        ss << "n_y" << n_y << "c_y" << c_y << "h_y" << h_y << "w_y" << w_y;

        ss << "xpk" << static_cast<int>(xdxDesc.IsPacked());
        ss << "ypk" << static_cast<int>(yDesc.IsPacked());
    }
    else
    {
        int n_y, c_y, h_y, w_y;
        int n_dy, c_dy, h_dy, w_dy;
        int n_dx, c_dx, h_dx, w_dx;

        std::tie(n_y, c_y, h_y, w_y)     = tien<4>(yDesc.GetLengths());
        std::tie(n_dy, c_dy, h_dy, w_dy) = tien<4>(dyDesc.GetLengths());
        std::tie(n_dx, c_dx, h_dx, w_dx) = tien<4>(xdxDesc.GetLengths());

        ss << "n_y" << n_y << "c_y" << c_y << "h_y" << h_y << "w_y" << w_y;
        ss << "n_dy" << n_dy << "c_dy" << c_dy << "h_dy" << h_dy << "w_dy" << w_dy;
        ss << "n_dx" << n_dx << "c_dx" << c_dx << "h_dx" << h_dx << "w_dx" << w_dx;

        ss << "ypk" << static_cast<int>(yDesc.IsPacked());
        ss << "dypk" << static_cast<int>(dyDesc.IsPacked());
        ss << "dxpk" << static_cast<int>(xdxDesc.IsPacked());
    }

    ss << "a" << alpha;
    ss << "b" << beta;
    ss << "algo" << static_cast<int>(algorithm);
    ss << "mode" << static_cast<int>(mode);

    return NetworkConfig{ss.str()};
}

} // namespace softmax

} // namespace miopen
