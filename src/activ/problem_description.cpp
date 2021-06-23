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

#include <miopen/activ/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace activ {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    // short cut for packed tensors and 2D tensors with stride != width
    const auto& x_lens = xDesc.GetLengths();

    const auto x_elem_sz = xDesc.GetElementSize();

    const auto x_width2D =
        ((x_lens.size() == 2) ? x_lens[1] : (x_lens.size() == 3) ? x_lens[2] : (x_lens.size() == 4)
                                                                                   ? x_lens[3]
                                                                                   : x_lens[4]);

    const auto height =
        (x_lens.size() == 2) ? x_lens[0] : (x_lens.size() == 3) ? x_lens[1] : (x_lens.size() == 4)
                                                                                  ? x_lens[2]
                                                                                  : x_lens[3];

    const auto packed = xDesc.IsPacked() && yDesc.IsPacked();

    const auto read_len = (packed) ? x_elem_sz : x_width2D;

    const auto read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    const auto MAP_RD    = read_len / read_unit;

    std::ostringstream ss;

    ss << "activ-";
    ss << ((packed) ? "11" : "10"); // + lite bit
    ss << xDesc.GetType();
    ss << activDesc.GetMode();
    ss << read_unit;
    ss << MAP_RD;
    ss << height;

    return NetworkConfig{ss.str()};
}

} // namespace activ

} // namespace miopen
