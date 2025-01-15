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

#include <miopen/matrix_diag/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace matrix_diag {

NetworkConfig ForwardProblemDescription::MakeNetworkConfig() const
{
    auto dtype      = diagDesc.GetType();
    auto diagSize   = diagDesc.GetElementSize();
    auto outputSize = outputDesc.GetElementSize();

    std::ostringstream ss;

    ss << "matrix_diag_fwd";
    ss << "dtype" << dtype;
    ss << "diagSize" << diagSize;
    ss << "outputSize" << outputSize;
    ss << "align" << align;

    return NetworkConfig{ss.str()};
}

} // namespace matrix_diag

namespace matrix_set_diag {

NetworkConfig ForwardProblemDescription::MakeNetworkConfig() const
{
    auto dtype      = diagDesc.GetType();
    auto diagSize   = diagDesc.GetElementSize();
    auto outputSize = outputDesc.GetElementSize();

    std::ostringstream ss;

    ss << "matrix_set_diag_fwd";
    ss << "dtype" << dtype;
    ss << "diagSize" << diagSize;
    ss << "outputSize" << outputSize;
    ss << "align" << align;

    return NetworkConfig{ss.str()};
}

} // namespace matrix_set_diag

} // namespace miopen
