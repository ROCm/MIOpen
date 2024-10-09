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

#include <miopen/kthvalue/problem_description.hpp>

#include <sstream>

namespace miopen {

namespace kthvalue {

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype = inputDesc.GetType();
    auto size        = inputDesc.GetElementSize();
    auto dim_size    = inputDesc.GetLengths()[dim];
    auto dim_stride  = inputDesc.GetStrides()[dim];
    int dim_num      = inputDesc.GetNumDims();
    auto output_size = size / dim_size;

    std::ostringstream ss;

    ss << "kthvalue_fwd";
    ss << "i_dtype" << input_dtype;
    ss << "dim_size" << dim_size;
    ss << "dim_num" << dim_num;
    ss << "dim_stride" << dim_stride;
    ss << "output_size" << output_size;

    return NetworkConfig{ss.str()};
}

} // namespace kthvalue

} // namespace miopen
