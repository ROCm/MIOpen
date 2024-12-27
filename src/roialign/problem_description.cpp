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
#include "miopen/names.hpp"
#include <sstream>

#include <miopen/roialign/problem_description.hpp>
// #include <miopen/names.hpp>

namespace miopen {

namespace roialign {

NetworkConfig FwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype         = inputDesc.GetType();
    auto input_lengths = inputDesc.GetLengths();

    std::ostringstream oss;

    oss << "RoIAlign_fwd";
    oss << "dtype" << dtype;
    oss << "input_lengths";
    for(auto length : input_lengths)
        oss << length << ',';
    // Add more information to the network config here
    // oss << "xdesc" << GetInputDesc();
    // oss << "ydesc" << GetOutputDesc();
    // oss << "rois" << GetRoisDesc();

    return NetworkConfig{oss.str()};
}

NetworkConfig BwdProblemDescription::MakeNetworkConfig() const
{
    auto dtype              = outputGradDesc.GetType();
    auto input_grad_lengths = inputGradDesc.GetLengths();

    std::ostringstream oss;

    oss << "RoIAlign_bwd";
    oss << "dtype" << dtype;
    oss << "input_grad_lengths";
    for(auto length : input_grad_lengths)
        oss << length << ',';
    // Add more information to the network config here
    // oss << "xdesc" << GetOutputGradDesc();
    // oss << "ydesc" << GetInputGradDesc();
    // oss << "rois" << GetRoisDesc();

    return NetworkConfig{oss.str()};
}

} // namespace roialign

} // namespace miopen
