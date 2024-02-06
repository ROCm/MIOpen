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

#include <miopen/layernorm/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace layernorm {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto dims         = xDesc.GetLengths();
    size_t outer_size = 1;
    size_t inner_size = 1;

    for(size_t i = 0ULL; i < dims.size(); ++i)
    {
        if(i < normalized_dim)
            outer_size *= dims[i];
        else
            inner_size *= dims[i];
    }

    auto dtype = xDesc.GetType();

    std::ostringstream ss;

    ss << "dtype" << dtype;
    ss << "normalized_dim" << normalized_dim;
    ss << "outer_size" << outer_size;
    ss << "inner_size" << inner_size;

    return NetworkConfig{ss.str()};
}

} // namespace layernorm

} // namespace miopen
