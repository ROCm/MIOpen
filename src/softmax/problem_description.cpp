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
    std::string network_config = "sfmfwd-";

    if(isForward)
    {
        int n_x, c_x, h_x, w_x;
        int n_y, c_y, h_y, w_y;

        std::tie(n_x, c_x, h_x, w_x) = tien<4>(xdxDesc.GetLengths());
        std::tie(n_y, c_y, h_y, w_y) = tien<4>(yDesc.GetLengths());

        network_config += "n_x" + std::to_string(n_x) + "c_x" + std::to_string(c_x) + "h_x" +
                          std::to_string(h_x) + "w_x" + std::to_string(w_x) +

                          "n_y" + std::to_string(n_y) + "c_y" + std::to_string(c_y) + "h_y" +
                          std::to_string(h_y) + "w_y" + std::to_string(w_y);

        network_config += "xpk" + std::to_string(static_cast<int>(xdxDesc.IsPacked())) + "ypk" +
                          std::to_string(static_cast<int>(yDesc.IsPacked()));
    }
    else
    {
        int n_y, c_y, h_y, w_y;
        int n_dy, c_dy, h_dy, w_dy;
        int n_dx, c_dx, h_dx, w_dx;

        std::tie(n_y, c_y, h_y, w_y)     = tien<4>(yDesc.GetLengths());
        std::tie(n_dy, c_dy, h_dy, w_dy) = tien<4>(dyDesc.GetLengths());
        std::tie(n_dx, c_dx, h_dx, w_dx) = tien<4>(xdxDesc.GetLengths());

        network_config += "n_y" + std::to_string(n_y) + "c_y" + std::to_string(c_y) + "h_y" +
                          std::to_string(h_y) + "w_y" + std::to_string(w_y) +

                          "n_dy" + std::to_string(n_dy) + "c_dy" + std::to_string(c_dy) + "h_dy" +
                          std::to_string(h_dy) + "w_dy" + std::to_string(w_dy) +

                          "n_dx" + std::to_string(n_dx) + "c_dx" + std::to_string(c_dx) + "h_dx" +
                          std::to_string(h_dx) + "w_dx" + std::to_string(w_dx) +

                          network_config +=
            "ypk" + std::to_string(static_cast<int>(yDesc.IsPacked())) + "dypk" +
            std::to_string(static_cast<int>(dyDesc.IsPacked())) + "dxpk" +
            std::to_string(static_cast<int>(xdxDesc.IsPacked()));
    }

    network_config += "a" + std::to_string(alpha) + "b" + std::to_string(beta) + "algo" +
                      std::to_string(static_cast<int>(algorithm)) + "mode" +
                      std::to_string(static_cast<int>(mode));

    return NetworkConfig{network_config};
}

} // namespace softmax

} // namespace miopen
