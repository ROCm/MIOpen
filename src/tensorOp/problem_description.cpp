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

#include <miopen/tensorOp/problem_description.hpp>
#include <miopen/names.hpp>
#include <miopen/float_equal.hpp>

namespace miopen {

namespace tensorOp {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::string ss;

    const auto& alens = aTensorDesc.GetLengths();
    const auto& blens = bTensorDesc.GetLengths();

    const auto& astrides = aTensorDesc.GetStrides();
    const auto& bstrides = bTensorDesc.GetStrides();
    const auto& cstrides = cTensorDesc.GetStrides();

    auto printDims = [&ss, dims = alens.size() - 1](const auto& dim) {
        for(uint32_t i = 0; i < dims; i++)
        {
            ss.append(std::to_string(dim[i]));
            ss += 'x';
        }
        ss += std::to_string(dim.back());
        ss += '-';
    };

    ss.reserve(1024);
    ss.append(std::string_view("TensorOp-"));
    ss += std::to_string(aTensorDesc.GetType());
    ss += '-';
    ss += std::to_string(tensorOp);
    ss += '-';

    printDims(alens);
    printDims(blens);
    printDims(astrides);
    printDims(bstrides);
    printDims(cstrides);

    ss += (float_equal(beta, 0.0f) ? '1' : '0');

    return NetworkConfig(std::move(ss));
}

} // namespace tensorOp

} // namespace miopen
