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

#include "miopen/datatype.hpp"
#include <miopen/where/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace where {

bool isContiguous(const TensorDescriptor& x)
{
    size_t s = 1;
    for(int i = x.GetSize() - 1; i >= 0; --i)
    {
        if(s != x.GetStrides()[i])
            return false;
        s *= x.GetLengths()[i];
    }
    return true;
}

NetworkConfig ForwardProblemDescription::MakeNetworkConfig() const
{
    auto input_numel = inputDesc.GetElementSize();
    auto other_numel = otherDesc.GetElementSize();
    auto cond_numel = conditionDesc.GetElementSize();
    auto input_dtype  = miopen::GetDataType(inputDesc.GetType());
    auto output_dtype = miopen::GetDataType(outputDesc.GetType());

    std::ostringstream ss;

    ss << "input_dtype" << input_dtype;
    ss << "output_dtype" << output_dtype;
    ss << "input_numel" << input_numel;
    ss << "other_numel" << other_numel;
    ss << "cond_numel" << cond_numel;
    // ss << IsAllPacked();

    return NetworkConfig{ss.str()};
}

NetworkConfig BackwardProblemDescription::MakeNetworkConfig() const
{
    auto input_grad_numel = inputGradDesc.GetElementSize();
    auto other_grad_numel = otherGradDesc.GetElementSize();
    auto cond_numel  = conditionDesc.GetElementSize();

    auto outputGrad_dtype     = miopen::GetDataType(outputGradDesc.GetType());
    auto inputGrad_dtype = miopen::GetDataType(inputGradDesc.GetType());

    std::ostringstream ss;

    ss << "inputGrad_dtype" << inputGrad_dtype;
    ss << "outputGrad_dtype" << outputGrad_dtype;
    ss << "inputGrad_numel" << input_grad_numel;
    ss << "otherGrad_numel" << other_grad_numel;
    ss << "cond_numel" << cond_numel;
    // ss << IsAllPacked();

    return NetworkConfig{ss.str()};
}

} // namespace where

} // namespace miopen
