
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
#ifndef GUARD_CPU_NLLLOSS_HPP
#define GUARD_CPU_NLLLOSS_HPP

#include "tensor_holder.hpp"
#include <miopen/tensor_view.hpp>

template <class T>
void cpu_nllloss_forward_4d(tensor<T> input,
                            tensor<int32_t> target,
                            tensor<T> weight,
                            tensor<T>& output,
                            int32_t ignore_index)
{
    auto dims = input.desc.GetLengths();
    size_t N  = dims[0];
    size_t C  = dims[1];
    size_t D1 = dims[2];
    size_t D2 = dims[3];

    auto I_tv = get_inner_expanded_tv_4d(input.desc);
    auto T_tv = get_inner_expanded_tv_3d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);
    auto O_tv = get_inner_expanded_tv_3d(output.desc);

    for(size_t i = 0; i < N * D1 * D2; i++)
    {
        uint64_t n[3];
        GET_NCD(n[0], n[1], n[2], i, O_tv);
        size_t target_index = TV3D_IDX(T_tv, n[0], n[1], n[2]);
        int32_t t           = target[target_index];
        size_t input_index  = TV4D_IDX(I_tv, n[0], t, n[1], n[2]);
        size_t weight_index = TV1D_IDX(W_tv, t);
        size_t output_index = TV3D_IDX(O_tv, n[0], n[1], n[2]);

        if(t < 0 || t == ignore_index || t >= C)
        {
            output[output_index] = static_cast<T>(0);
        }
        else
        {
            output[output_index] =
                static_cast<T>(-1.0f) * weight[weight_index] * input[input_index];
        }
    }
}

template <class T>
void cpu_nllloss_backward_4d(tensor<T>& input_grad,
                             tensor<int32_t> target,
                             tensor<T> weight,
                             tensor<T> output_grad,
                             int32_t ignore_index)
{
    auto dims = input_grad.desc.GetLengths();
    size_t N  = dims[0];
    size_t C  = dims[1];
    size_t D1 = dims[2];
    size_t D2 = dims[3];

    auto I_tv = get_inner_expanded_tv_4d(input_grad.desc);
    auto T_tv = get_inner_expanded_tv_3d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);
    auto O_tv = get_inner_expanded_tv_3d(output_grad.desc);

    for(size_t i = 0; i < N * D1 * D2; i++)
    {
        uint64_t n[3];
        GET_NCD(n[0], n[1], n[2], i, O_tv);
        size_t target_index      = TV3D_IDX(T_tv, n[0], n[1], n[2]);
        int32_t t                = target[target_index];
        size_t input_grad_index  = TV4D_IDX(I_tv, n[0], t, n[1], n[2]);
        size_t weight_index      = TV1D_IDX(W_tv, t);
        size_t output_grad_index = TV3D_IDX(O_tv, n[0], n[1], n[2]);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<T>(0);
        }
        else
        {
            input_grad[input_grad_index] =
                static_cast<T>(-1.0f) * weight[weight_index] * output_grad[output_grad_index];
        }
    }
}
#endif
