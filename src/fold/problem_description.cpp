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

#include <miopen/fold/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace fold {

NetworkConfig UnfoldFwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype  = inputDesc.GetType();
    auto output_dtype = outputDesc.GetType();
    auto size         = inputDesc.GetElementSize();
    auto in_dims      = inputDesc.GetLengths();

    std::ostringstream ss;

    ss << "Unfold_fwd";
    ss << "i_dtype" << input_dtype;
    ss << "o_dtype" << output_dtype;
    ss << "size" << size;
    ss << "in_dims";
    for(auto val : in_dims)
    {
        ss << "_" << val;
    }
    ss << "kernel_size_" << kernel_size[0] << "_" << kernel_size[1];
    ss << "stride_" << stride[0] << "_" << stride[1];
    ss << "padding_" << padding[0] << "_" << padding[1];
    ss << "dilation_" << dilation[0] << "_" << dilation[1];

    return NetworkConfig{ss.str()};
}

NetworkConfig UnfoldBwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype  = dinputDesc.GetType();
    auto output_dtype = doutputDesc.GetType();
    auto size         = dinputDesc.GetElementSize();
    auto in_dims      = dinputDesc.GetLengths();

    std::ostringstream ss;

    ss << "Unfold_bwd";
    ss << "i_dtype" << input_dtype;
    ss << "o_dtype" << output_dtype;
    ss << "size" << size;
    ss << "in_grad_dims";
    for(auto val : in_dims)
    {
        ss << "_" << val;
    }
    ss << "kernel_size_" << kernel_size[0] << "_" << kernel_size[1];
    ss << "stride_" << stride[0] << "_" << stride[1];
    ss << "padding_" << padding[0] << "_" << padding[1];
    ss << "dilation_" << dilation[0] << "_" << dilation[1];

    return NetworkConfig{ss.str()};
}

NetworkConfig FoldFwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype  = inputDesc.GetType();
    auto output_dtype = outputDesc.GetType();
    auto size         = inputDesc.GetElementSize();
    auto in_dims      = inputDesc.GetLengths();
    auto out_dims     = outputDesc.GetLengths();

    std::ostringstream ss;

    ss << "Fold_fwd";
    ss << "i_dtype" << input_dtype;
    ss << "o_dtype" << output_dtype;
    ss << "size" << size;
    ss << "in_dims";
    for(auto val : in_dims)
    {
        ss << "_" << val;
    }
    ss << "out_dims";
    for(auto val : out_dims)
    {
        ss << "_" << val;
    }
    ss << "kernel_size_" << kernel_size[0] << "_" << kernel_size[1];
    ss << "stride_" << stride[0] << "_" << stride[1];
    ss << "padding_" << padding[0] << "_" << padding[1];
    ss << "dilation_" << dilation[0] << "_" << dilation[1];

    return NetworkConfig{ss.str()};
}

NetworkConfig FoldBwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype  = dinputDesc.GetType();
    auto output_dtype = doutputDesc.GetType();
    auto size         = dinputDesc.GetElementSize();
    auto in_dims      = dinputDesc.GetLengths();
    auto out_dims     = doutputDesc.GetLengths();

    std::ostringstream ss;

    ss << "Fold_bwd";
    ss << "i_dtype" << input_dtype;
    ss << "o_dtype" << output_dtype;
    ss << "size" << size;
    ss << "in_grad_dims";
    for(auto val : in_dims)
    {
        ss << "_" << val;
    }
    ss << "out_grad_dims";
    for(auto val : out_dims)
    {
        ss << "_" << val;
    }
    ss << "kernel_size_" << kernel_size[0] << "_" << kernel_size[1];
    ss << "stride_" << stride[0] << "_" << stride[1];
    ss << "padding_" << padding[0] << "_" << padding[1];
    ss << "dilation_" << dilation[0] << "_" << dilation[1];

    return NetworkConfig{ss.str()};
}

} // namespace fold

} // namespace miopen
