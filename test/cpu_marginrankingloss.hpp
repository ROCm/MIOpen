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
#pragma once

#include "tensor_holder.hpp"
#include <miopen/tensor_view.hpp>

template <class T>
void cpu_marginrankingloss_reduced_forward_5d(tensor<T> input1,
                                              tensor<T> input2,
                                              tensor<T> target,
                                              tensor<T>& output,
                                              float margin,
                                              float divisor)
{
    tensor_view_5d_t I1_tv = get_inner_expanded_tv_5d(input1.desc);
    tensor_view_5d_t I2_tv = get_inner_expanded_tv_5d(input2.desc);
    tensor_view_5d_t T_tv  = get_inner_expanded_tv_5d(target.desc);
    tensor_view_5d_t O_tv  = get_inner_expanded_tv_5d(output.desc);
    size_t tensor_size     = target.desc.GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < tensor_size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, T_tv)
        size_t I1idx = TV5D_IDX(I1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t I2idx = TV5D_IDX(I2_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx  = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Oidx  = TV5D_IDX(O_tv, n[0], n[1], n[2], n[3], n[4]);

        output[Oidx] = -target[Tidx] * (input1[I1idx] - input2[I2idx]) + static_cast<T>(margin);
        if(output[Oidx] < 0)
            output[Oidx] = 0.0f;
        output[Oidx] /= static_cast<T>(divisor);
    }
}

template <class T>
void cpu_marginrankingloss_reduced_backward_5d(tensor<T> input1,
                                               tensor<T> input2,
                                               tensor<T> target,
                                               tensor<T> outGrad,
                                               tensor<T>& in1Grad,
                                               tensor<T>& in2Grad,
                                               float margin,
                                               float divisor)
{
    tensor_view_5d_t I1_tv  = get_inner_expanded_tv_5d(input1.desc);
    tensor_view_5d_t I2_tv  = get_inner_expanded_tv_5d(input2.desc);
    tensor_view_5d_t T_tv   = get_inner_expanded_tv_5d(target.desc);
    tensor_view_5d_t dO_tv  = get_inner_expanded_tv_5d(outGrad.desc);
    tensor_view_5d_t dI1_tv = get_inner_expanded_tv_5d(in1Grad.desc);
    tensor_view_5d_t dI2_tv = get_inner_expanded_tv_5d(in2Grad.desc);
    size_t tensor_size      = target.desc.GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < tensor_size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, T_tv)
        size_t I1idx  = TV5D_IDX(I1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t I2idx  = TV5D_IDX(I2_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx   = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dOidx  = TV5D_IDX(dO_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dI1idx = TV5D_IDX(dI1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dI2idx = TV5D_IDX(dI2_tv, n[0], n[1], n[2], n[3], n[4]);

        T t = -target[Tidx] * (input1[I1idx] - input2[I2idx]) + static_cast<T>(margin);

        if(t < 0)
        {
            in2Grad[dI2idx] = 0.0f;
            in1Grad[dI1idx] = 0.0f;
        }
        else
        {
            in1Grad[dI1idx] = -target[Tidx] * outGrad[dOidx] / divisor;
            in2Grad[dI2idx] = target[Tidx] * outGrad[dOidx] / divisor;
        }
    }
}

template <class T>
void cpu_marginrankingloss_unreduced_forward_5d(
    tensor<T> input1, tensor<T> input2, tensor<T> target, tensor<T>& output, float margin)
{
    tensor_view_5d_t I1_tv = get_inner_expanded_tv_5d(input1.desc);
    tensor_view_5d_t I2_tv = get_inner_expanded_tv_5d(input2.desc);
    tensor_view_5d_t T_tv  = get_inner_expanded_tv_5d(target.desc);
    tensor_view_5d_t O_tv  = get_inner_expanded_tv_5d(output.desc);
    size_t tensor_size     = target.desc.GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < tensor_size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, T_tv)
        size_t I1idx = TV5D_IDX(I1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t I2idx = TV5D_IDX(I2_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx  = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Oidx  = TV5D_IDX(O_tv, n[0], n[1], n[2], n[3], n[4]);

        output[Oidx] = -target[Tidx] * (input1[I1idx] - input2[I2idx]) + static_cast<T>(margin);
        if(output[Oidx] < 0)
            output[Oidx] = 0.0f;
    }
}

template <class T>
void cpu_marginrankingloss_unreduced_backward_5d(tensor<T> input1,
                                                 tensor<T> input2,
                                                 tensor<T> target,
                                                 tensor<T> outGrad,
                                                 tensor<T>& in1Grad,
                                                 tensor<T>& in2Grad,
                                                 float margin)
{
    tensor_view_5d_t I1_tv  = get_inner_expanded_tv_5d(input1.desc);
    tensor_view_5d_t I2_tv  = get_inner_expanded_tv_5d(input2.desc);
    tensor_view_5d_t T_tv   = get_inner_expanded_tv_5d(target.desc);
    tensor_view_5d_t dO_tv  = get_inner_expanded_tv_5d(outGrad.desc);
    tensor_view_5d_t dI1_tv = get_inner_expanded_tv_5d(in1Grad.desc);
    tensor_view_5d_t dI2_tv = get_inner_expanded_tv_5d(in2Grad.desc);
    size_t tensor_size      = target.desc.GetElementSize();
    size_t n[5];

    for(size_t idx = 0; idx < tensor_size; ++idx)
    {
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], idx, T_tv)
        size_t I1idx  = TV5D_IDX(I1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t I2idx  = TV5D_IDX(I2_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx   = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dOidx  = TV5D_IDX(dO_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dI1idx = TV5D_IDX(dI1_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t dI2idx = TV5D_IDX(dI2_tv, n[0], n[1], n[2], n[3], n[4]);

        T t = -target[Tidx] * (input1[I1idx] - input2[I2idx]) + static_cast<T>(margin);

        if(t < 0)
        {
            in1Grad[dI1idx] = 0.0f;
            in2Grad[dI2idx] = 0.0f;
        }
        else
        {
            in1Grad[dI1idx] = -target[Tidx] * outGrad[dOidx];
            in2Grad[dI2idx] = target[Tidx] * outGrad[dOidx];
        }
    }
}
