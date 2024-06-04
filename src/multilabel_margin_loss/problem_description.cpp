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

#include <miopen/multilabel_margin_loss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace multilabel_margin_loss {

NetworkConfig MultilabelMarginLossFwdProblemDescription::MakeNetworkConfig() const
{
    auto ilength    = iDesc.GetLengths();
    auto tlength    = tDesc.GetLengths();
    auto i_size = std::accumulate(
        ilength.begin(), ilength.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    auto idtype = iDesc.GetType();
    auto tdtype = tDesc.GetType();

    std::ostringstream ss;
    ss << "fwd";
    ss << "i_dtype" << idtype;
    ss << "t_dtype" << tdtype;
    ss << "i_size" << i_size;
    return NetworkConfig{ss.str()};
}

NetworkConfig MultilabelMarginLossUnreducedFwdProblemDescription::MakeNetworkConfig() const
{
    auto ilength    = iDesc.GetLengths();
    auto tlength    = tDesc.GetLengths();
    auto i_size = std::accumulate(
        ilength.begin(), ilength.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    auto idtype = iDesc.GetType();
    auto tdtype = tDesc.GetType();

    std::ostringstream ss;
    ss << "fwd";
    ss << "unreduced";
    ss << "i_dtype" << idtype;
    ss << "t_dtype" << tdtype;
    ss << "i_size" << i_size;
    return NetworkConfig{ss.str()};
}

NetworkConfig MultilabelMarginLossBwdProblemDescription::MakeNetworkConfig() const
{
    auto ilength    = iDesc.GetLengths();
    auto tlength    = tDesc.GetLengths();
    auto i_size = std::accumulate(
        ilength.begin(), ilength.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    auto idtype = iDesc.GetType();
    auto tdtype = tDesc.GetType();

    std::ostringstream ss;
    ss << "bwd";
    ss << "i_dtype" << idtype;
    ss << "t_dtype" << tdtype;
    ss << "i_size" << i_size;
    return NetworkConfig{ss.str()};
}

NetworkConfig MultilabelMarginLossUnreducedBwdProblemDescription::MakeNetworkConfig() const
{
    auto ilength    = iDesc.GetLengths();
    auto tlength    = tDesc.GetLengths();
    auto i_size = std::accumulate(
        ilength.begin(), ilength.end(), static_cast<size_t>(1), std::multiplies<size_t>());

    auto idtype = iDesc.GetType();
    auto tdtype = tDesc.GetType();

    std::ostringstream ss;
    ss << "bwd";
    ss << "unreduced";
    ss << "i_dtype" << idtype;
    ss << "t_dtype" << tdtype;
    ss << "i_size" << i_size;
    return NetworkConfig{ss.str()};
}

} // namespace multilabel_margin_loss

} // namespace miopen
