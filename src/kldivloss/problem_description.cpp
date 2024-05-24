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
#include <miopen/kldivloss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace kldivloss {

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

NetworkConfig UnreducedProblemDescription::MakeNetworkConfig() const
{
    size_t numel       = GetNtotal();
    size_t num_batches = inputDesc.GetLengths()[0];
    size_t num_dims    = inputDesc.GetSize();
    bool is_log_target = GetLogTarget();
    auto input_dtype   = inputDesc.GetType();
    auto Si            = inputDesc.GetStrides();
    auto St            = targetDesc.GetStrides();
    auto So            = outputDesc.GetStrides();

    std::ostringstream ss;

    ss << "kldivloss_unreduced";
    ss << "is_fwd" << is_fwd;
    ss << "log_target" << is_log_target;
    ss << "input_dtype" << input_dtype;
    ss << "numel" << numel;
    ss << "num_dims" << num_dims;
    ss << "num_batches" << num_batches;
    ss << "input_stride" << Si;
    ss << "target_stride" << St;
    ss << "output_stride" << So;

    return NetworkConfig{ss.str()};
}

NetworkConfig ReducedProblemDescription::MakeNetworkConfig() const
{
    size_t numel       = GetNtotal();
    size_t num_batches = inputDesc.GetLengths()[0];
    size_t num_dims    = inputDesc.GetSize();
    bool is_log_target = GetLogTarget();
    auto input_dtype   = inputDesc.GetType();
    auto Si            = inputDesc.GetStrides();
    auto St            = targetDesc.GetStrides();
    auto So            = outputDesc.GetStrides();

    std::ostringstream ss;

    ss << "kldivloss_reduced";
    ss << "is_fwd" << is_fwd;
    ss << "divisor" << divisor;
    ss << "log_target" << is_log_target;
    ss << "input_dtype" << input_dtype;
    ss << "numel" << numel;
    ss << "num_dims" << num_dims;
    ss << "num_batches" << num_batches;
    ss << "input_stride" << Si;
    ss << "target_stride" << St;
    ss << "output_stride" << So;

    return NetworkConfig{ss.str()};
}

} // namespace kldivloss

} // namespace miopen
