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

#include <miopen/getitem/problem_description.hpp>
#include <miopen/datatype.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace getitem {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dy_dims     = dyDesc.GetLengths();
    auto input_dtype = dyDesc.GetType();
    auto error_dtype = errorDesc.GetType();

    auto input_size =
        std::accumulate(dy_dims.begin(), dy_dims.end(), 1ULL, std::multiplies<size_t>());

    std::ostringstream ss;

    ss << "getitembwd";
    ss << "input_size" << input_size;
    ss << "input_dtype" << input_dtype;
    ss << "error_dtype" << error_dtype;
    ss << "indexCount" << indexCount;

    for(int i = 0; i < indexCount; ++i)
    {
        if(i == 0)
            ss << "indexs_size";
        const auto& index_dims = (*indexDescs)[i].GetLengths();
        auto index_size        = std::accumulate(
            index_dims.begin(), index_dims.begin(), 1ULL, std::multiplies<size_t>());
        ss << index_size << "_";
    }

    return NetworkConfig{ss.str()};
}

} // namespace getitem

} // namespace miopen
