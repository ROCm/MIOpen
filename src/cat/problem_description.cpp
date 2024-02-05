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

#include <miopen/cat/problem_description.hpp>
#include <miopen/datatype.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace cat {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    int32_t fusion_size = 2;
    while(fusion_size < xCount)
    {
        fusion_size *= 2;
    }

    size_t max_x_dim_size = 0;
    for(int i = 0; i < xCount; i++)
    {
        auto xlength   = xDescs[i]->GetLengths();
        max_x_dim_size = std::max(max_x_dim_size, xlength[dim]);
    }

    auto ylength    = yDesc.GetLengths();
    auto outer_size = std::accumulate(
        ylength.begin(), ylength.begin() + dim, static_cast<size_t>(1), std::multiplies<size_t>());
    auto stride         = yDesc.GetStrides()[dim];
    auto dtype          = yDesc.GetType();
    auto data_size      = get_data_size(dtype);
    auto max_inner_size = max_x_dim_size * stride * data_size / sizeof(short4);

    std::ostringstream ss;

    ss << "catfwd" << fusion_size;
    ss << "max_inner_size" << max_inner_size;
    ss << "outer_size" << outer_size;

    return NetworkConfig{ss.str()};
}

} // namespace cat

} // namespace miopen
