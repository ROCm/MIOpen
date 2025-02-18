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

#ifndef GUARD_CPU_PAD_CONSTANT_HPP
#define GUARD_CPU_PAD_CONSTANT_HPP

#include "miopen/tensor.hpp"
#include <cstddef>
#include <sys/types.h>
#include "../src/kernels/tensor_view_5d.hpp"
#include "miopen/tensor_view_5d.hpp"

template <class T>
void cpu_pad_constant_fwd(const T* input,
                          T* output,
                          miopen::TensorDescriptor* input_desc,
                          miopen::TensorDescriptor* output_desc,
                          std::vector<int64_t> padding_vec,
                          T value)
{
    auto input_tv  = get_inner_expanded_tv(*input_desc);
    auto output_tv = get_inner_expanded_tv(*output_desc);

    padding_5d_t padding;
    memset(padding.val, 0, sizeof(padding.val));
    for(auto i = 0; i < padding_vec.size(); i++)
        padding.val[10 - padding_vec.size() + i] = padding_vec[i];

    int64_t o[5];

    for(size_t gid = 0; gid != output_desc->GetElementSize(); ++gid)
    {
        bool flag = true;
        getNCDHW(o, gid, output_tv.size);
        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] - padding.val[2 * i];
            flag *= (o[i] >= 0) && (o[i] < input_tv.size[i]);
        }

        auto val =
            flag ? get5DValueAt(input, input_tv.stride, o[0], o[1], o[2], o[3], o[4]) : (T)value;

        set5DValueAt(output, output_tv, gid, val);
    }
}
#endif

template <class T>
void cpu_pad_constant_bwd(T* input_grad,
                          T* backward_output,
                          miopen::TensorDescriptor* backward_output_desc,
                          miopen::TensorDescriptor* input_grad_desc,
                          std::vector<int64_t> padding_vec)
{
    auto output_tv = get_inner_expanded_tv(*backward_output_desc);
    auto input_tv  = get_inner_expanded_tv(*input_grad_desc);

    padding_5d_t padding;
    memset(padding.val, 0, sizeof(padding.val));
    for(auto i = 0; i < padding_vec.size(); i++)
        padding.val[10 - padding_vec.size() + i] = padding_vec[i];

    int64_t o[5];
    size_t backward_output_size = backward_output_desc->GetElementSize();

    for(size_t gid = 0; gid < backward_output_size; ++gid)
    {
        bool flag = true;
        getNCDHW(o, gid, output_tv.size);

        for(int i = 0; i < 5; i++)
        {
            o[i] = o[i] + padding.val[2 * i];
            flag *= (o[i] >= 0) && (o[i] < input_tv.size[i]);
        }

        T val = flag ? get5DValueAt<T>(input_grad, input_tv.stride, o[0], o[1], o[2], o[3], o[4])
                     : (T)0;
        set5DValueAt(backward_output, output_tv, gid, val);
    }
}
