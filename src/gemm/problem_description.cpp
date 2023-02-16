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

#include <miopen/gemm/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace gemm {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    // short cut for packed tensors and 2D tensors with stride != width
    const auto& A_lens = ADesc.GetLengths();

    const auto A_elem_sz = ADesc.GetElementSize();

    const auto A_width2D =
        ((A_lens.size() == 2)
             ? A_lens[1]
             : (A_lens.size() == 3) ? A_lens[2] : (A_lens.size() == 4) ? A_lens[3] : A_lens[4]);

    const auto height =
        (A_lens.size() == 2)
            ? A_lens[0]
            : (A_lens.size() == 3) ? A_lens[1] : (A_lens.size() == 4) ? A_lens[2] : A_lens[3];

    const auto packed = ADesc.IsPacked() && BDesc.IsPacked();

    const auto read_len = (packed) ? A_elem_sz : A_width2D;

    const auto read_unit = (read_len % 4 == 0) ? 4 : (read_len % 2 == 0) ? 2 : 1;
    const auto MAP_RD    = read_len / read_unit;

    std::ostringstream ss;

    ss << "gemm-";

    ss << ((packed) ? "11" : "10"); 
    ss << ADesc.GetType();
    ss << read_unit;
    ss << MAP_RD;
    ss << height;

    return NetworkConfig{ss.str()};
}

} // namespace gemm

} // namespace miopen
