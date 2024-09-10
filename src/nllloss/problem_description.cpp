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

#include <cstddef>
#include <miopen/nllloss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace nllloss {

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    size_t numel       = GetNtotal();
    size_t num_batches = inputDesc.GetLengths()[0];
    size_t num_classes = GetC();
    size_t num_dims    = inputDesc.GetNumDims();

    auto input_dtype = inputDesc.GetType();

    std::ostringstream ss;

    ss << "nllloss_reduce";
    ss << "is_fwd" << is_fwd;
    ss << "input_dtype" << input_dtype;
    ss << "reduce_mode" << reduction;
    ss << "numel" << numel;
    ss << "num_dims" << num_dims;
    ss << "num_batches" << num_batches;
    ss << "num_classes" << num_classes;

    return NetworkConfig{ss.str()};
}

} // namespace nllloss

} // namespace miopen
