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

#include <miopen/sigmoidfocalloss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace sigmoidfocalloss {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y)
{
    if(x.GetNumDims() != y.GetNumDims())
        return false;
    for(int32_t i = 0; i < x.GetNumDims(); ++i)
    {
        if(x.GetLengths()[i] != y.GetLengths()[i])
            return false;
    }
    return true;
}

NetworkConfig SigmoidFocalLossBwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype  = inputDesc.GetType();
    auto target_dtype = targetDesc.GetType();
    auto size         = inputDesc.GetElementSize();
    auto dim_num      = inputDesc.GetNumDims();

    std::ostringstream ss;

    ss << "sfl_bwd";
    ss << "reduction" << reduction;
    ss << "i_dtype" << input_dtype;
    ss << "t_dtype" << target_dtype;
    ss << "dim_num" << dim_num;
    ss << "size" << size;

    return NetworkConfig{ss.str()};
}

NetworkConfig SigmoidFocalLossFwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype  = inputDesc.GetType();
    auto target_dtype = targetDesc.GetType();
    auto size         = inputDesc.GetElementSize();
    auto dim_num      = inputDesc.GetNumDims();

    std::ostringstream ss;

    ss << "sfl_fwd";
    ss << "reduction" << reduction;
    ss << "i_dtype" << input_dtype;
    ss << "t_dtype" << target_dtype;
    ss << "dim_num" << dim_num;
    ss << "size" << size;

    return NetworkConfig{ss.str()};
}

} // namespace sigmoidfocalloss

} // namespace miopen
