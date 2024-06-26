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
#ifndef GUARD_CPU_INTERPOLATE_HPP
#define GUARD_CPU_INTERPOLATE_HPP

#include "tensor_holder.hpp"
#include <miopen/interpolate/utils.hpp>

template <class T>
void cpu_interpolate_linear_forward(const tensor<T> input,
                                    tensor<T>& output,
                                    const size_t nelems,
                                    const float* scale_factors,
                                    const bool align_corners)
{
    auto I_tv = get_inner_expanded_tv<3>(input.desc);
    auto O_tv = get_inner_expanded_tv<3>(output.desc);

    size_t num_batches = I_tv.size[0];
    size_t num_class   = I_tv.size[1];
}

template <class T>
void cpu_interpolate_linear_backward(tensor<T> output_grad,
                                     tensor<T> backprop,
                                     tensor<T> input,
                                     tensor<T>& input_grad,
                                     tensor<T>& target_grad,
                                     bool input_grad_out,
                                     bool target_grad_out)
{
    auto dO_tv = get_inner_expanded_tv_1d(output_grad.desc);
    auto B_tv  = get_inner_expanded_tv_2d(backprop.desc);
    auto I_tv  = get_inner_expanded_tv_2d(input.desc);

    size_t num_batches = I_tv.size[0];
    size_t num_class   = I_tv.size[1];
}

#endif // GUARD_CPU_INTERPOLATE_HPP