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
#include <miopen/embedding/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace embedding {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    bool is_fwd       = (direction == Direction::Forward);
    auto input_dtype  = inputDesc.GetType();
    auto weight_dtype = weightDesc.GetType();
    auto output_dtype = outputDesc.GetType();
    auto input_dims   = inputDesc.GetLengths();
    auto weight_dims  = weightDesc.GetLengths();
    auto kernel       = "embedding";
    auto fwd          = is_fwd ? "fwd" : "bwd";

    std::ostringstream ss;

    ss << kernel << fwd;
    ss << "dtype" << input_dtype << weight_dtype << output_dtype;
    ss << "input";
    for(auto dim : input_dims)
        ss << dim << ",";

    ss << "weight";
    for(auto dim : weight_dims)
        ss << dim << ",";
    /*
        if(IsAllPacked())
            ss << "packed";
    */
    if(!is_fwd)
        ss << scale_grad_by_freq << deterministic_mode;

    return NetworkConfig{ss.str()};
}

} // namespace embedding

} // namespace miopen
