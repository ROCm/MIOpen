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

#include <miopen/instancenorm/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace instancenorm {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y)
{
    if(x.GetSize() != y.GetSize())
        return false;
    for(int32_t i = 0; i < x.GetSize(); ++i)
    {
        if(x.GetLengths()[i] != y.GetLengths()[i])
            return false;
    }
    return true;
}

NetworkConfig InstanceNormFwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype  = inputDesc.GetType();
    auto output_dtype = outputDesc.GetType();
    auto size         = inputDesc.GetElementSize();
    auto dim_num      = inputDesc.GetSize();
    auto mean_in_size = meanInDesc.GetElementSize();
    auto mean_out_size = meanOutDesc.GetElementSize();
    auto mean_var_size = meanVarDesc.GetElementSize();

    std::ostringstream ss;

    ss << "instnorm_fwd";
    ss << "i_dtype" << input_dtype;
    ss << "o_dtype" << output_dtype;
    ss << "dim_num" << dim_num;
    ss << "size" << size;
    ss << "mean_in_num" << mean_in_size;
    ss << "mean_out_num" << mean_out_size;
    ss << "mean_var_num" << mean_var_size;
    ss << "use_input_stats" << (useInputStats ? "true" : "false");

    return NetworkConfig{ss.str()};
}

} // namespace instancenorm

} // namespace miopen
