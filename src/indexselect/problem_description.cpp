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

#include <miopen/datatype.hpp>
#include <miopen/indexselect/problem_description.hpp>
#include <miopen/names.hpp>

namespace miopen {

namespace indexselect {

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    ss << "indexselectfwd";
    auto dtype = inputDesc.GetType();
    ss << "dtype" << dtype;
    ss << "output_numel" << outputDesc.GetElementSize();

    return NetworkConfig{ss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    ss << "indexselectbwd";
    auto dtype = inputGradDesc.GetType();
    ss << "dtype" << dtype;
    ss << "input_numel" << inputGradDesc.GetElementSize();
    ss << "output_numel" << outputGradDesc.GetElementSize();

    return NetworkConfig{ss.str()};
}

} // namespace indexselect

} // namespace miopen
