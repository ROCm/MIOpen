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

#include <miopen/reduce/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace reduce {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    auto xlength = xDesc.GetLengths();
    std::vector<std::size_t> outputlength;
    if((reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MIN) ||
       (reduceExtremeOp == MIOPEN_REDUCE_EXTREME_MAX) ||
       (reduceExtremeOp == MIOPEN_REDUCE_CALCULATION_SUM))
        outputlength = yDesc.GetLengths();
    else
        outputlength = indiceDesc.GetLengths();

    auto size         = xlength[dim];
    auto output_numel = std::accumulate(outputlength.begin(),
                                        outputlength.end(),
                                        static_cast<size_t>(1),
                                        std::multiplies<size_t>());
    auto dtype        = xDesc.GetType();

    std::ostringstream ss;

    ss << "dtype" << dtype;
    ss << "dim" << dim;
    ss << "size" << size;
    ss << "output_numel" << output_numel;
    ss << "reduceExtremeOp" << reduceExtremeOp;

    return NetworkConfig{ss.str()};
}

} // namespace reduce

} // namespace miopen
