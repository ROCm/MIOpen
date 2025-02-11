/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#include <sstream>

#include <miopen/generate_random_bit_mask/problem_description.hpp>

namespace miopen {
namespace generate_random_bit_mask {

NetworkConfig InitPRNGStateProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    ss << "init_prng_state";

    ss << "state_size_in_bytes: " << stateSizeInBytes;

    return NetworkConfig{ss.str()};
}

NetworkConfig ProblemDescription::MakeNetworkConfig() const
{
    std::ostringstream ss;
    ss << "generate_random_bit_mask";
    ss << "state_size_in_bytes: " << stateSizeInBytes;
    ss << "mask_size_in_bytes: " << maskSizeInBytes;
    ss << "p: " << p;

    return NetworkConfig{ss.str()};
}

} // namespace generate_random_bit_mask
} // namespace miopen
