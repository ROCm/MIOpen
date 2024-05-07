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
#include <miopen/glu/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace glu {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    switch(direction)
    {
    case Direction::Forward: return MakeForwardNetworkConfig();
    case Direction::Backward: return MakeBackwardNetworkConfig();
    default: MIOPEN_THROW(miopenStatusInternalError);
    }
}

NetworkConfig ProblemDescription::MakeForwardNetworkConfig() const
{
    auto inputlength = inputDesc.GetLengths();

    auto input_numel = std::accumulate(
        inputlength.begin(), inputlength.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    auto input_dtype  = miopen::GetDataType(inputDesc.GetType());
    auto output_dtype = miopen::GetDataType(outputDesc.GetType());

    std::ostringstream ss;

    ss << "input_dtype" << input_dtype;
    ss << "output_dtype" << output_dtype;
    ss << "dim" << dim;
    ss << "input_numel" << input_numel;
    ss << IsAllPacked();

    return NetworkConfig{ss.str()};
}

NetworkConfig ProblemDescription::MakeBackwardNetworkConfig() const
{
    auto inputlength = inputDesc.GetLengths();

    auto input_numel = std::accumulate(
        inputlength.begin(), inputlength.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    auto input_dtype     = miopen::GetDataType(inputDesc.GetType());
    auto inputGrad_dtype = miopen::GetDataType(inputGradDesc.GetType());

    std::ostringstream ss;

    ss << "input_dtype" << input_dtype;
    ss << "inputGrad_dtype" << inputGrad_dtype;
    ss << "dim" << dim;
    ss << "input_numel" << input_numel;
    ss << IsAllPacked();

    return NetworkConfig{ss.str()};
}

} // namespace glu

} // namespace miopen
