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
#ifndef GUARD_CPU_MULTILABEL_MARGIN_LOSS_HPP
#define GUARD_CPU_MULTILABEL_MARGIN_LOSS_HPP

#include "miopen/tensor.hpp"
#include "tensor_holder.hpp"
#include <algorithm>
#include <cstddef>

template <class TIO, class TT>
void cpu_multilabel_margin_loss_forward_2d(tensor<TIO> input,
                                      tensor<TT> target,
                                      tensor<TIO>& workspace,
                                      tensor<TIO>& ref_output,
                                      float divisor = 1)
{
    auto idims = input.desc.GetLengths();
    auto tdims = target.desc.GetLengths();
    auto istrides = input.desc.GetStrides();
    auto tstrides = target.desc.GetStrides();
    auto input_size = std::accumulate(idims.begin(), idims.end(), 1L, std::multiplies<int64_t>());

    auto N = idims[0];
    auto C = idims[1];
    auto ws = std::vector<char>(input_size, static_cast<char>(0.0));
    // Compute loss
    for(size_t idx = 0; idx < N; ++idx)
    {
        auto n = idx;
        for (size_t c = 0; c < C; c++) 
        {
            ws[n * C + c] = 0;
        }
        
        for (size_t c = 0; c < C; c++) 
        {
            int is_target_idx = 0;
            for (size_t i = 0; i < C; i++)
            {
                size_t T_at_n_i = target[tstrides[1] * i + tstrides[0] * n];
                if (T_at_n_i == -1) break;
                if (T_at_n_i == c) 
                {
                    is_target_idx = 1;
                    break;
                }
            }
            if (is_target_idx)
            {
                ws[n * C + c] = 1;
            }
        }
        float loss = 0.0f;

        for (size_t ct = 0; ct < C; ct++)
        {
            size_t T_at_n_ct = target[tstrides[1] * ct + tstrides[0] * n];
            if (T_at_n_ct == -1) break;
            for (size_t ci = 0; ci < C; ci++)
            {
                if (ws[n * C + ci] == 0)
                {
                    float t = 1.0f - static_cast<float>(input[istrides[1] * T_at_n_ct + istrides[0] * n]) - static_cast<float>(input[istrides[1] * ci + istrides[0] * n]);
                    t /= C;
                    loss += t >= 0 ? t : 0.0f;
                }
            }
        }

        workspace[n] = static_cast<TIO>(loss / divisor);
    }

    // Reduce loss
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = N;
    size_t _size         = N;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            TIO shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? workspace[offset_a + i + j] : static_cast<TIO>(0.0f);
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                ref_output[0] = shared[0];
            else
                workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class TIO, class TT>
void cpu_multilabel_margin_loss_unreduced_forward_2d(tensor<TIO> input,
                                      tensor<TT> target,
                                      tensor<TIO>& workspace,
                                      tensor<TIO>& ref_output)
{
    auto idims = input.desc.GetLengths();
    auto tdims = target.desc.GetLengths();
    auto istrides = input.desc.GetStrides();
    auto tstrides = target.desc.GetStrides();
    auto ostrides = ref_output.desc.GetStrides();
    auto input_size = std::accumulate(idims.begin(), idims.end(), 1L, std::multiplies<int64_t>());

    auto N = idims[0];
    auto C = idims[1];
    auto ws = std::vector<char>(input_size, static_cast<char>(0.0));
    // Compute loss
    for(size_t idx = 0; idx < N; ++idx)
    {
        auto n = idx;
        for (size_t c = 0; c < C; c++) 
        {
            ws[n * C + c] = 0;
        }
        
        for (size_t c = 0; c < C; c++) 
        {
            int is_target_idx = 0;
            for (size_t i = 0; i < C; i++)
            {
                size_t T_at_n_i = target[tstrides[1] * i + tstrides[0] * n];
                if (T_at_n_i == -1) break;
                if (T_at_n_i == c) 
                {
                    is_target_idx = 1;
                    break;
                }
            }
            if (is_target_idx)
            {
                ws[n * C + c] = 1;
            }
        }
        float loss = 0.0f;

        for (size_t ct = 0; ct < C; ct++)
        {
            size_t T_at_n_ct = target[tstrides[1] * ct + tstrides[0] * n];
            if (T_at_n_ct == -1) break;
            for (size_t ci = 0; ci < C; ci++)
            {
                if (ws[n * C + ci] == 0)
                {
                    float t = 1.0f - static_cast<float>(input[istrides[1] * T_at_n_ct + istrides[0] * n]) - static_cast<float>(input[istrides[1] * ci + istrides[0] * n]);
                    t /= C;
                    loss += t >= 0 ? t : 0.0f;
                }
            }
        }

        ref_output[ostrides[0] * n] = static_cast<TIO>(loss);
    }
}

template <class TIO, class TT>
void cpu_multilabel_margin_loss_backward_2d(tensor<TIO> input,
                                      tensor<TT> target,
                                      tensor<TIO>& workspace,
                                      tensor<TIO>& dO,
                                      tensor<TIO>& ref_dI,
                                      float divisor = 1)
{
    auto idims = input.desc.GetLengths();
    auto tdims = target.desc.GetLengths();
    auto istrides = input.desc.GetStrides();
    auto tstrides = target.desc.GetStrides();
    auto input_size = std::accumulate(idims.begin(), idims.end(), 1L, std::multiplies<int64_t>());

    auto dO_strides = dO.desc.GetStrides();
    auto ref_dI_strides = ref_dI.desc.GetStrides();

    auto N = idims[0];
    auto C = idims[1];
    auto ws = std::vector<char>(input_size, static_cast<char>(0.0));
    // Compute loss
    for(size_t idx = 0; idx < N; ++idx)
    {
        auto n = idx;
        for (size_t c = 0; c < C; c++) 
        {
            ws[n * C + c] = 0;
            ref_dI[(ref_dI_strides[1] * c) + (ref_dI_strides[0] * n)] = 0.0f;
        }
        
        for (size_t c = 0; c < C; c++) 
        {
            int is_target_idx = 0;
            for (size_t i = 0; i < C; i++)
            {
                size_t T_at_n_i = target[tstrides[1] * i + tstrides[0] * n];
                if (T_at_n_i == -1) break;
                if (T_at_n_i == c) 
                {
                    is_target_idx = 1;
                    break;
                }
            }
            if (is_target_idx)
            {
                ws[n * C + c] = 1;
            }
        }
        float out_grad = dO[dO_strides[0] * 0];
        float delta = 1.0f / C * out_grad  / divisor;

        for (size_t ct = 0; ct < C; ct++)
        {
            size_t T_at_n_ct = target[tstrides[1] * ct + tstrides[0] * n];
            if (T_at_n_ct == -1) break;
            for (size_t ci = 0; ci < C; ci++)
            {
                if (ws[n * C + ci] == 0)
                {
                    float t = 1.0f - static_cast<float>(input[istrides[1] * T_at_n_ct + istrides[0] * n]) - static_cast<float>(input[istrides[1] * ci + istrides[0] * n]);
                    if (t >= 0)
                    {
                        float x = static_cast<float>(ref_dI[(ref_dI_strides[1] * ci) + (ref_dI_strides[0] * n)]) + delta;
                        ref_dI[(ref_dI_strides[1] * ci) + (ref_dI_strides[0] * n)] = static_cast<TIO>(x);
                        float y = static_cast<float>(ref_dI[(ref_dI_strides[1] * T_at_n_ct) + (ref_dI_strides[0] * n)]) - delta;
                        ref_dI[(ref_dI_strides[1] * T_at_n_ct) + (ref_dI_strides[0] * n)] = static_cast<TIO>(y);
                    }
                }
            }
        }
    }
}

template <class TIO, class TT>
void cpu_multilabel_margin_loss_unreduced_backward_2d(tensor<TIO> input,
                                      tensor<TT> target,
                                      tensor<TIO>& workspace,
                                      tensor<TIO>& dO,
                                      tensor<TIO>& ref_dI)
{
    auto idims = input.desc.GetLengths();
    auto tdims = target.desc.GetLengths();
    auto istrides = input.desc.GetStrides();
    auto tstrides = target.desc.GetStrides();
    auto input_size = std::accumulate(idims.begin(), idims.end(), 1L, std::multiplies<int64_t>());

    auto dO_strides = dO.desc.GetStrides();
    auto ref_dI_strides = ref_dI.desc.GetStrides();

    auto N = idims[0];
    auto C = idims[1];
    auto ws = std::vector<char>(input_size, static_cast<char>(0.0));
    // Compute loss
    for(size_t idx = 0; idx < N; ++idx)
    {
        auto n = idx;
        for (size_t c = 0; c < C; c++) 
        {
            ws[n * C + c] = 0;
            ref_dI[(ref_dI_strides[1] * c) + (ref_dI_strides[0] * n)] = 0.0f;
        }
        
        for (size_t c = 0; c < C; c++) 
        {
            int is_target_idx = 0;
            for (size_t i = 0; i < C; i++)
            {
                size_t T_at_n_i = target[tstrides[1] * i + tstrides[0] * n];
                if (T_at_n_i == -1) break;
                if (T_at_n_i == c) 
                {
                    is_target_idx = 1;
                    break;
                }
            }
            if (is_target_idx)
            {
                ws[n * C + c] = 1;
            }
        }
        float out_grad = dO[dO_strides[0] * n];
        float delta = 1.0f / C * out_grad;

        for (size_t ct = 0; ct < C; ct++)
        {
            size_t T_at_n_ct = target[tstrides[1] * ct + tstrides[0] * n];
            if (T_at_n_ct == -1) break;
            for (size_t ci = 0; ci < C; ci++)
            {
                if (ws[n * C + ci] == 0)
                {
                    float t = 1.0f - static_cast<float>(input[istrides[1] * T_at_n_ct + istrides[0] * n]) - static_cast<float>(input[istrides[1] * ci + istrides[0] * n]);
                    if (t >= 0)
                    {
                        float x = static_cast<float>(ref_dI[(ref_dI_strides[1] * ci) + (ref_dI_strides[0] * n)]) + delta;
                        ref_dI[(ref_dI_strides[1] * ci) + (ref_dI_strides[0] * n)] = static_cast<TIO>(x);
                        float y = static_cast<float>(ref_dI[(ref_dI_strides[1] * T_at_n_ct) + (ref_dI_strides[0] * n)]) - delta;
                        ref_dI[(ref_dI_strides[1] * T_at_n_ct) + (ref_dI_strides[0] * n)] = static_cast<TIO>(y);
                    }
                }
            }
        }
    }
}
#endif
