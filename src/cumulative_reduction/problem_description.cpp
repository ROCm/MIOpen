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

#include <miopen/cumulative_reduction/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace cumulative_reduction {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y)
{
    if(x.GetSize() != y.GetSize())
        return false;
    for(int i = 0; i < x.GetSize(); ++i)
    {
        if(x.GetLengths()[i] != y.GetLengths()[i])
            return false;
    }
    return true;
}

NetworkConfig ForwardProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype  = inputDesc.GetType();
    auto output_dtype = outputDesc.GetType();
    auto size         = inputDesc.GetElementSize();
    auto inner_size   = inputDesc.GetLengths()[dim];
    auto outer_size   = size / inner_size;

    std::ostringstream ss;

    ss << "idtype" << input_dtype;
    ss << "odtype" << output_dtype;
    ss << "outer" << outer_size;
    ss << "inner" << inner_size;
    ss << "op" << cumOp;
    ss << "packed" << IsAllPacked();
    ss << "dimstride1" << IsAllDimStride1();

    return NetworkConfig{ss.str()};
}

} // namespace cumulative_reduction

} // namespace miopen
