
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
void cpu_nllloss_unreduce_forward(tensor<T> input,
                                  tensor<int32_t> target,
                                  tensor<T> weight,
                                  tensor<T>& output,
                                  int32_t ignore_index)
{
    auto num_dims = input.desc.GetSize();
    if(num_dims == 2)
    {
        cpu_nllloss_unreduce_forward_2d(input, target, weight, output, ignore_index);
    }
    else if(num_dims < 5)
    {
        cpu_nllloss_unreduce_forward_4d(input, target, weight, output, ignore_index);
    }
    else if(num_dims < 6)
    {
        cpu_nllloss_unreduce_forward_5d(input, target, weight, output, ignore_index);
    }
}

template <class T>
void cpu_nllloss_reduce_forward_5d(tensor<T> input,
                                   tensor<int32_t> target,
                                   tensor<T> weight,
                                   tensor<T>& output,
                                   tensor<T>& workspace,
                                   int32_t ignore_index,
                                   float divisor)
{
    auto dims  = input.desc.GetLengths();
    auto numel = target.desc.GetElementSize();
    size_t C   = dims[1];

    auto I_tv = get_inner_expanded_tv_5d(input.desc);
    auto T_tv = get_inner_expanded_tv_4d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);

    for(size_t i = 0; i < numel; i++)
    {
        uint64_t n[4];
        GET_NCDH(n[0], n[1], n[2], n[3], i, T_tv);
        size_t target_index = TV4D_IDX(T_tv, n[0], n[1], n[2], n[3]);
        int32_t t           = target[target_index];
        size_t input_index  = TV5D_IDX(I_tv, n[0], t, n[1], n[2], n[3]);
        size_t weight_index = TV1D_IDX(W_tv, t);

        if(t < 0 || t == ignore_index || t >= C)
        {
            workspace[i] = static_cast<T>(0.0f);
        }
        else
        {
            T w = weight[weight_index];

            T input_value = input[input_index];
            T d           = !std::isnan(divisor) ? static_cast<T>(divisor) : static_cast<T>(1.0f);
            workspace[i]  = (-w * input_value) / d;
        }
    }

    auto reduce_size     = numel;
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = reduce_size;
    size_t _size         = reduce_size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            T shared[local_size];
            par_ford(local_size)([&](size_t j) {
                shared[j] = i + j < _size ? workspace[offset_a + i + j] : static_cast<T>(0.0f);
            });
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                ford(local_size)([&](size_t j) {
                    if(j < offset)
                        shared[j] += shared[j + offset];
                });
            if(_size <= local_size)
                output[0] = shared[0];
            else
                workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class T>
void cpu_nllloss_unreduce_backward(tensor<T>& input_grad,
                                   tensor<int32_t> target,
                                   tensor<T> weight,
                                   tensor<T> output_grad,
                                   int32_t ignore_index)
{
    auto num_dims = input_grad.desc.GetSize();
    if(num_dims == 2)
    {
        cpu_nllloss_unreduce_backward_2d(input_grad, target, weight, output_grad, ignore_index);
    }
    else if(num_dims < 5)
    {
        cpu_nllloss_unreduce_backward_4d(input_grad, target, weight, output_grad, ignore_index);
    }
    else if(num_dims < 6)
    {
        cpu_nllloss_unreduce_backward_5d(input_grad, target, weight, output_grad, ignore_index);
    }
}

template <class T>
void cpu_nllloss_reduce_backward(tensor<T>& input_grad,
                                 tensor<int32_t> target,
                                 tensor<T> weight,
                                 tensor<T> output_grad,
                                 int32_t ignore_index,
                                 float divisor)
{
    auto num_dims = input_grad.desc.GetSize();
    if(num_dims == 2)
    {
        cpu_nllloss_reduce_backward_2d(
            input_grad, target, weight, output_grad, ignore_index, divisor);
    }
    else if(num_dims < 6)
    {
        cpu_nllloss_reduce_backward_5d(
            input_grad, target, weight, output_grad, ignore_index, divisor);
    }
}

template <class T>
void cpu_nllloss_unreduce_forward_2d(tensor<T> input,
                                     tensor<int32_t> target,
                                     tensor<T> weight,
                                     tensor<T>& output,
                                     int32_t ignore_index)
{
    auto dims = input.desc.GetLengths();
    size_t C  = dims[1];

    auto I_tv = get_inner_expanded_tv_2d(input.desc);
    auto T_tv = get_inner_expanded_tv_1d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);
    auto O_tv = get_inner_expanded_tv_1d(output.desc);

    for(size_t i = 0; i < target.desc.GetElementSize(); i++)
    {
        size_t target_index = TV1D_IDX(T_tv, i);
        int32_t t           = target[target_index];
        size_t input_index  = TV2D_IDX(I_tv, i, t);
        size_t weight_index = TV1D_IDX(W_tv, t);
        size_t output_index = TV1D_IDX(O_tv, i);

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
void cpu_nllloss_unreduce_forward_4d(tensor<T> input,
                                     tensor<int32_t> target,
                                     tensor<T> weight,
                                     tensor<T>& output,
                                     int32_t ignore_index)
{
    auto dims = input.desc.GetLengths();
    size_t C  = dims[1];

    auto I_tv = get_inner_expanded_tv_4d(input.desc);
    auto T_tv = get_inner_expanded_tv_3d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);
    auto O_tv = get_inner_expanded_tv_3d(output.desc);

    for(size_t i = 0; i < target.desc.GetElementSize(); i++)
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
void cpu_nllloss_unreduce_forward_5d(tensor<T> input,
                                     tensor<int32_t> target,
                                     tensor<T> weight,
                                     tensor<T>& output,
                                     int32_t ignore_index)
{
    auto dims = input.desc.GetLengths();
    size_t C  = dims[1];

    auto I_tv = get_inner_expanded_tv_5d(input.desc);
    auto T_tv = get_inner_expanded_tv_4d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);
    auto O_tv = get_inner_expanded_tv_4d(output.desc);

    for(size_t i = 0; i < target.desc.GetElementSize(); i++)
    {
        uint64_t n[4];
        GET_NCDH(n[0], n[1], n[2], n[3], i, O_tv);
        size_t target_index = TV4D_IDX(T_tv, n[0], n[1], n[2], n[3]);
        int32_t t           = target[target_index];
        size_t input_index  = TV5D_IDX(I_tv, n[0], t, n[1], n[2], n[3]);
        size_t weight_index = TV1D_IDX(W_tv, t);
        size_t output_index = TV4D_IDX(O_tv, n[0], n[1], n[2], n[3]);

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
void cpu_nllloss_unreduce_backward_2d(tensor<T>& input_grad,
                                      tensor<int32_t> target,
                                      tensor<T> weight,
                                      tensor<T> output_grad,
                                      int32_t ignore_index)
{
    auto dims = input_grad.desc.GetLengths();
    size_t C  = dims[1];

    auto I_tv = get_inner_expanded_tv_2d(input_grad.desc);
    auto T_tv = get_inner_expanded_tv_1d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);
    auto O_tv = get_inner_expanded_tv_1d(output_grad.desc);

    for(size_t i = 0; i < target.desc.GetElementSize(); i++)
    {
        size_t target_index      = TV1D_IDX(T_tv, i);
        int32_t t                = target[target_index];
        size_t input_grad_index  = TV2D_IDX(I_tv, i, t);
        size_t weight_index      = TV1D_IDX(W_tv, t);
        size_t output_grad_index = TV1D_IDX(O_tv, i);

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

template <class T>
void cpu_nllloss_unreduce_backward_4d(tensor<T>& input_grad,
                                      tensor<int32_t> target,
                                      tensor<T> weight,
                                      tensor<T> output_grad,
                                      int32_t ignore_index)
{
    auto dims = input_grad.desc.GetLengths();
    size_t C  = dims[1];

    auto I_tv = get_inner_expanded_tv_4d(input_grad.desc);
    auto T_tv = get_inner_expanded_tv_3d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);
    auto O_tv = get_inner_expanded_tv_3d(output_grad.desc);

    for(size_t i = 0; i < target.desc.GetElementSize(); i++)
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

template <class T>
void cpu_nllloss_unreduce_backward_5d(tensor<T>& input_grad,
                                      tensor<int32_t> target,
                                      tensor<T> weight,
                                      tensor<T> output_grad,
                                      int32_t ignore_index)
{
    auto dims = input_grad.desc.GetLengths();
    size_t C  = dims[1];

    auto I_tv = get_inner_expanded_tv_5d(input_grad.desc);
    auto T_tv = get_inner_expanded_tv_4d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);
    auto O_tv = get_inner_expanded_tv_4d(output_grad.desc);

    for(size_t i = 0; i < target.desc.GetElementSize(); i++)
    {
        uint64_t n[4];
        GET_NCDH(n[0], n[1], n[2], n[3], i, O_tv);
        size_t target_index      = TV4D_IDX(T_tv, n[0], n[1], n[2], n[3]);
        int32_t t                = target[target_index];
        size_t input_grad_index  = TV5D_IDX(I_tv, n[0], t, n[1], n[2], n[3]);
        size_t weight_index      = TV1D_IDX(W_tv, t);
        size_t output_grad_index = TV4D_IDX(O_tv, n[0], n[1], n[2], n[3]);

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

template <class T>
void cpu_nllloss_reduce_backward_2d(tensor<T>& input_grad,
                                    tensor<int32_t> target,
                                    tensor<T> weight,
                                    tensor<T> output_grad,
                                    int32_t ignore_index,
                                    float divisor)
{
    auto dims = input_grad.desc.GetLengths();
    size_t C  = dims[1];

    auto I_tv = get_inner_expanded_tv_2d(input_grad.desc);
    auto T_tv = get_inner_expanded_tv_1d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);

    for(size_t i = 0; i < target.desc.GetElementSize(); i++)
    {
        size_t target_index     = TV1D_IDX(T_tv, i);
        int32_t t               = target[target_index];
        size_t input_grad_index = TV2D_IDX(I_tv, i, t);
        size_t weight_index     = TV1D_IDX(W_tv, t);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<T>(0);
        }
        else
        {
            input_grad[input_grad_index] =
                (static_cast<T>(-1.0f) * weight[weight_index] * output_grad[0]) /
                static_cast<T>(divisor);
        }
    }
}

template <class T>
void cpu_nllloss_reduce_backward_5d(tensor<T>& input_grad,
                                    tensor<int32_t> target,
                                    tensor<T> weight,
                                    tensor<T> output_grad,
                                    int32_t ignore_index,
                                    float divisor)
{
    auto dims = input_grad.desc.GetLengths();
    size_t C  = dims[1];

    auto I_tv = get_inner_expanded_tv_5d(input_grad.desc);
    auto T_tv = get_inner_expanded_tv_4d(target.desc);
    auto W_tv = get_inner_expanded_tv_1d(weight.desc);

    for(size_t i = 0; i < target.desc.GetElementSize(); i++)
    {
        uint64_t n[4];
        GET_NCDH(n[0], n[1], n[2], n[3], i, T_tv);
        size_t target_index     = TV4D_IDX(T_tv, n[0], n[1], n[2], n[3]);
        int32_t t               = target[target_index];
        size_t input_grad_index = TV5D_IDX(I_tv, n[0], t, n[1], n[2], n[3]);
        size_t weight_index     = TV1D_IDX(W_tv, t);

        if(t < 0 || t == ignore_index || t >= C)
        {
            input_grad[input_grad_index] = static_cast<T>(0);
        }
        else
        {
            input_grad[input_grad_index] =
                (static_cast<T>(-1.0f) * weight[weight_index] * output_grad[0]) /
                static_cast<T>(divisor);
        }
    }
}
#endif
