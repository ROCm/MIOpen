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
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_cosineembeddingloss_unreduced_forward_2d(
    tensor<T> input1, tensor<T> input2, tensor<int32_t> target, tensor<T>& output, float margin)
{
    auto I1_tv = get_inner_expanded_tv<2>(input1.desc);
    auto I2_tv = get_inner_expanded_tv<2>(input2.desc);
    auto T_tv  = get_inner_expanded_tv<1>(target.desc);
    auto O_tv  = get_inner_expanded_tv<1>(output.desc);

    size_t N = input1.desc.GetLengths()[0], D = input1.desc.GetLengths()[1];
    for(size_t n = 0; n < N; ++n)
    {
        size_t Tidx = T_tv.get_tensor_view_idx({n});
        int32_t t   = target[Tidx];

        float cos_term = 0.0f;
        float norm1 = 0.0f, norm2 = 0.0f;

        for(size_t d = 0; d < D; ++d)
        {
            size_t I1idx = I1_tv.get_tensor_view_idx({n, d});
            size_t I2idx = I2_tv.get_tensor_view_idx({n, d});
            cos_term += static_cast<float>(input1[I1idx]) * static_cast<float>(input2[I2idx]);
            norm1 += static_cast<float>(input1[I1idx]) * static_cast<float>(input1[I1idx]);
            norm2 += static_cast<float>(input2[I2idx]) * static_cast<float>(input2[I2idx]);
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        cos_term /= norm1 * norm2;

        size_t Oidx = O_tv.get_tensor_view_idx({n});

        if(t == 1)
        {
            output[Oidx] = static_cast<T>(1.0f - cos_term);
        }
        else
        {
            output[Oidx] = static_cast<T>(std::max(0.0f, cos_term - margin));
        }
    }
}

template <class T>
void cpu_cosineembeddingloss_reduced_forward_2d(tensor<T> input1,
                                                tensor<T> input2,
                                                tensor<int32_t> target,
                                                tensor<T>& output,
                                                tensor<T>& workspace,
                                                float margin,
                                                float divisor)
{
    auto I1_tv = get_inner_expanded_tv<2>(input1.desc);
    auto I2_tv = get_inner_expanded_tv<2>(input2.desc);
    auto T_tv  = get_inner_expanded_tv<1>(target.desc);

    size_t N = input1.desc.GetLengths()[0], D = input1.desc.GetLengths()[1];
    for(size_t n = 0; n < N; ++n)
    {
        size_t Tidx    = T_tv.get_tensor_view_idx({n});
        int32_t t      = target[Tidx];
        float cos_term = 0.0f;
        float norm1 = 0.0f, norm2 = 0.0f;
        for(size_t d = 0; d < D; ++d)
        {
            size_t I1idx = I1_tv.get_tensor_view_idx({n, d});
            size_t I2idx = I2_tv.get_tensor_view_idx({n, d});
            cos_term += static_cast<float>(input1[I1idx]) * static_cast<float>(input2[I2idx]);
            norm1 += static_cast<float>(input1[I1idx]) * static_cast<float>(input1[I1idx]);
            norm2 += static_cast<float>(input2[I2idx]) * static_cast<float>(input2[I2idx]);
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        cos_term /= norm1 * norm2;

        float loss = 0.0f;
        if(t == 1)
            loss = 1.0f - cos_term;
        else
            loss = std::max(0.0f, cos_term - margin);

        workspace[n] = static_cast<T>(loss / divisor);
    }

    auto reduce_size     = N;
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = reduce_size;
    size_t _size         = reduce_size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            float shared[local_size];
            for(auto j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? static_cast<float>(workspace[offset_a + i + j]) : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(auto j = 0; j < local_size; ++j)
                    if(j < offset)
                        shared[j] += shared[j + offset];
            if(_size <= local_size)
                output[0] = static_cast<T>(shared[0]);
            else
                workspace[offset_b + i / local_size] = static_cast<T>(shared[0]);
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class T>
void cpu_cosineembeddingloss_unreduced_backward_2d(tensor<T> input1,
                                                   tensor<T> input2,
                                                   tensor<int32_t> target,
                                                   tensor<T> output_grad,
                                                   tensor<T>& input1_grad,
                                                   tensor<T>& input2_grad,
                                                   float margin,
                                                   bool input1_grad_out,
                                                   bool input2_grad_out)
{
    auto I1_tv  = get_inner_expanded_tv<2>(input1.desc);
    auto I2_tv  = get_inner_expanded_tv<2>(input2.desc);
    auto T_tv   = get_inner_expanded_tv<1>(target.desc);
    auto dO_tv  = get_inner_expanded_tv<1>(output_grad.desc);
    auto dI1_tv = get_inner_expanded_tv<2>(input1_grad.desc);
    auto dI2_tv = get_inner_expanded_tv<2>(input2_grad.desc);

    size_t N = input1.desc.GetLengths()[0], D = input1.desc.GetLengths()[1];
    for(size_t n = 0; n < N; ++n)
    {
        for(size_t d = 0; d < D; ++d)
        {
            if(input1_grad_out)
            {
                size_t dI1idx       = dI1_tv.get_tensor_view_idx({n, d});
                input1_grad[dI1idx] = static_cast<T>(0.0f);
            }
            if(input2_grad_out)
            {
                size_t dI2idx       = dI2_tv.get_tensor_view_idx({n, d});
                input2_grad[dI2idx] = static_cast<T>(0.0f);
            }
        }

        size_t Tidx = T_tv.get_tensor_view_idx({n});
        int32_t t   = target[Tidx];

        float cos_term = 0.0f;
        float norm1 = 0.0f, norm2 = 0.0f;

        for(size_t d = 0; d < D; ++d)
        {
            size_t I1idx = I1_tv.get_tensor_view_idx({n, d});
            size_t I2idx = I2_tv.get_tensor_view_idx({n, d});
            cos_term += static_cast<float>(input1[I1idx]) * static_cast<float>(input2[I2idx]);
            norm1 += static_cast<float>(input1[I1idx]) * static_cast<float>(input1[I1idx]);
            norm2 += static_cast<float>(input2[I2idx]) * static_cast<float>(input2[I2idx]);
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        cos_term /= norm1 * norm2;

        for(size_t d = 0; d < D; ++d)
        {
            size_t I1idx  = I1_tv.get_tensor_view_idx({n, d});
            size_t I2idx  = I2_tv.get_tensor_view_idx({n, d});
            size_t dOidx  = dO_tv.get_tensor_view_idx({n});
            size_t dI1idx = dI1_tv.get_tensor_view_idx({n, d});
            size_t dI2idx = dI2_tv.get_tensor_view_idx({n, d});
            float i1      = static_cast<float>(input1[I1idx]);
            float i2      = static_cast<float>(input2[I2idx]);
            float og      = static_cast<float>(output_grad[dOidx]);
            if(t == 1)
            {
                if(input1_grad_out)
                {
                    float input1_grad_value =
                        -(i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1));
                    input1_grad[dI1idx] = static_cast<T>(input1_grad_value * og);
                }
                if(input2_grad_out)
                {
                    float input2_grad_value =
                        -(i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2));
                    input2_grad[dI2idx] = static_cast<T>(input2_grad_value * og);
                }
            }
            else
            {
                if(cos_term - margin < 0.0f)
                    continue;
                if(input1_grad_out)
                {
                    float input1_grad_value =
                        i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1);
                    input1_grad[dI1idx] = static_cast<T>(input1_grad_value * og);
                }
                if(input2_grad_out)
                {
                    float input2_grad_value =
                        i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2);
                    input2_grad[dI2idx] = static_cast<T>(input2_grad_value * og);
                }
            }
        }
    }
}

template <class T>
void cpu_cosineembeddingloss_reduced_backward_2d(tensor<T> input1,
                                                 tensor<T> input2,
                                                 tensor<int32_t> target,
                                                 tensor<T> output_grad,
                                                 tensor<T>& input1_grad,
                                                 tensor<T>& input2_grad,
                                                 float margin,
                                                 float divisor,
                                                 bool input1_grad_out,
                                                 bool input2_grad_out)
{
    auto I1_tv  = get_inner_expanded_tv<2>(input1.desc);
    auto I2_tv  = get_inner_expanded_tv<2>(input2.desc);
    auto T_tv   = get_inner_expanded_tv<1>(target.desc);
    auto dO_tv  = get_inner_expanded_tv<1>(output_grad.desc);
    auto dI1_tv = get_inner_expanded_tv<2>(input1_grad.desc);
    auto dI2_tv = get_inner_expanded_tv<2>(input2_grad.desc);

    size_t N = input1.desc.GetLengths()[0], D = input1.desc.GetLengths()[1];
    for(size_t n = 0; n < N; ++n)
    {
        for(size_t d = 0; d < D; ++d)
        {
            if(input1_grad_out)
            {
                size_t dI1idx       = dI1_tv.get_tensor_view_idx({n, d});
                input1_grad[dI1idx] = static_cast<T>(0.0f);
            }
            if(input2_grad_out)
            {
                size_t dI2idx       = dI2_tv.get_tensor_view_idx({n, d});
                input2_grad[dI2idx] = static_cast<T>(0.0f);
            }
        }

        size_t Tidx = T_tv.get_tensor_view_idx({n});
        int32_t t   = target[Tidx];

        float cos_term = 0.0f;
        float norm1 = 0.0f, norm2 = 0.0f;

        for(size_t d = 0; d < D; ++d)
        {
            size_t I1idx = I1_tv.get_tensor_view_idx({n, d});
            size_t I2idx = I2_tv.get_tensor_view_idx({n, d});
            cos_term += static_cast<float>(input1[I1idx]) * static_cast<float>(input2[I2idx]);
            norm1 += static_cast<float>(input1[I1idx]) * static_cast<float>(input1[I1idx]);
            norm2 += static_cast<float>(input2[I2idx]) * static_cast<float>(input2[I2idx]);
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        cos_term /= norm1 * norm2;
        size_t dOidx = dO_tv.get_tensor_view_idx({0});
        float og     = static_cast<float>(output_grad[dOidx]);

        for(size_t d = 0; d < D; ++d)
        {
            size_t I1idx = I1_tv.get_tensor_view_idx({n, d});
            size_t I2idx = I2_tv.get_tensor_view_idx({n, d});

            float i1              = static_cast<float>(input1[I1idx]);
            float i2              = static_cast<float>(input2[I2idx]);
            float input1_grad_val = 0.0f;
            float input2_grad_val = 0.0f;

            if(t == 1)
            {
                if(input1_grad_out)
                {
                    input1_grad_val = -(i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1));
                }
                if(input2_grad_out)
                {
                    input2_grad_val = -(i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2));
                }
            }
            else
            {
                if(cos_term - margin < 0.0f)
                    continue;
                if(input1_grad_out)
                {
                    input1_grad_val = i2 / (norm1 * norm2) - cos_term * i1 / (norm1 * norm1);
                }
                if(input2_grad_out)
                {
                    input2_grad_val = i1 / (norm1 * norm2) - cos_term * i2 / (norm2 * norm2);
                }
            }
            size_t dI1idx = dI1_tv.get_tensor_view_idx({n, d});
            if(input1_grad_out)
            {
                input1_grad[dI1idx] = static_cast<T>(input1_grad_val * og / divisor);
            }
            size_t dI2idx = dI2_tv.get_tensor_view_idx({n, d});
            if(input2_grad_out)
            {
                input2_grad[dI2idx] = static_cast<T>(input2_grad_val * og / divisor);
            }
        }
    }
}
