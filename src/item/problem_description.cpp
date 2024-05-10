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

#include <miopen/item/problem_description.hpp>
#include <miopen/datatype.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace item {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dx_dims         = dxDesc.GetLengths();
    auto index_dims      = (*indexDescs)[0].GetLengths();
    auto input_dtype     = dyDesc.GetType();
    auto output_dtype    = dxDesc.GetType();
    auto dim_info_offset = indexCount > 0 ? indexCount * index_dims[0] : 0;
    auto start_dim       = dims[0];

    std::vector<int32_t> output_dims(dimCount);
    for(int32_t i = 0; i < dimCount; i++)
    {
        output_dims[i] = static_cast<int32_t>(dx_dims[dims[i]]);
    }
    std::ostringstream ss;

    ss << "getitembwd";
    ss << "input_dtype" << input_dtype;
    ss << "output_dtype" << output_dtype;
    ss << "indexCount" << indexCount;
    ss << "offset" << offset;
    ss << "dim_info_offset" << dim_info_offset;
    ss << "index_dims";
    for(int32_t i = 0; i < dimCount; i++)
        ss << dims[i] << "_";
    ss << "slices";
    for(int32_t i = 0; i < sliceCount; i++)
        ss << slices[i] << "_";
    ss << "output_dims";
    for(auto output_dim : output_dims)
        ss << output_dim << "_";
    ss << "start_dim" << start_dim;

    return NetworkConfig{ss.str()};
}

} // namespace item

} // namespace miopen
