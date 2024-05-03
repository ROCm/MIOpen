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
    auto numel = targetDesc.GetElementSize();

    size_t num_batches = N;
    size_t num_classes = C;

    auto input_dtype  = weightDesc.GetType();

    std::ostringstream ss;

    ss << "input_dtype" << input_dtype;
    ss << "numel" << numel;
    ss << "num_batches" << num_batches;
    ss << "num_classes" << num_classes;
    ss << "D1" << D1;
    ss << "D2" << D2;
    ss << "is_bwd" << is_bwd;

    return NetworkConfig{ss.str()};
}

} // namespace nllloss

} // namespace miopen
