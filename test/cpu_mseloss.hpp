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

#ifndef GUARD_CPU_MSE_LOSS_HPP
#define GUARD_CPU_MSE_LOSS_HPP

#include "miopen/tensor.hpp"
#include "miopen/tensor_view_utils.hpp"

template <class T>
void cpu_mseloss(miopen::TensorDescriptor inputDesc,
                 miopen::TensorDescriptor targetDesc,
                 miopen::TensorDescriptor outputDesc,
                 const T* input,
                 const T* target,
                 T* output,
                 float divisor)
{
    tensor_view_t<5> I_tv = miopen::get_inner_expanded_tv<5>(inputDesc);
    tensor_view_t<5> T_tv = miopen::get_inner_expanded_tv<5>(targetDesc);

    int64_t gid = 0;
    *output     = 0;

    while(true)
    {
        size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
        size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
        size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
        size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

        if(!(n0 < I_tv.size[0]))
            break;

        size_t Iidx = I_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        size_t Tidx = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

        *output +=
            static_cast<T>((input[Iidx] - target[Tidx]) * (input[Iidx] - target[Tidx]) / divisor);

        ++gid;
    }
}

template <class T>
void cpu_mseloss_backward(miopen::TensorDescriptor inputDesc,
                          miopen::TensorDescriptor targetDesc,
                          miopen::TensorDescriptor outputDesc,
                          miopen::TensorDescriptor inputGradDesc,
                          miopen::TensorDescriptor targetGradDesc,
                          const T* input,
                          const T* target,
                          const T* output,
                          T* input_grad,
                          T* target_grad,
                          float divisor)
{
    tensor_view_t<5> I_tv  = miopen::get_inner_expanded_tv<5>(inputDesc);
    tensor_view_t<5> T_tv  = miopen::get_inner_expanded_tv<5>(targetDesc);
    tensor_view_t<5> IG_tv = miopen::get_inner_expanded_tv<5>(inputGradDesc);
    tensor_view_t<5> TG_tv = miopen::get_inner_expanded_tv<5>(targetGradDesc);

    int64_t gid = 0;

    while(true)
    {
        size_t n0123 = gid / I_tv.size[4], n4 = gid % I_tv.size[4];
        size_t n012 = n0123 / I_tv.size[3], n3 = n0123 % I_tv.size[3];
        size_t n01 = n012 / I_tv.size[2], n2 = n012 % I_tv.size[2];
        size_t n0 = n01 / I_tv.size[1], n1 = n01 % I_tv.size[1];

        if(!(n0 < I_tv.size[0]))
            break;

        size_t Iidx  = I_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        size_t Tidx  = T_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        size_t IGidx = IG_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});
        size_t TGidx = TG_tv.get_tensor_view_idx({n0, n1, n2, n3, n4});

        T grad = static_cast<T>(2.0f) * (input[Iidx] - target[Tidx]) / static_cast<T>(divisor) *
                 output[0];

        if(input_grad != nullptr)
        {
            input_grad[IGidx] = grad;
        }

        if(target_grad != nullptr)
        {
            target_grad[TGidx] = -grad;
        }
        ++gid;
    }
}
#endif
