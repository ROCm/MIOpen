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

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>

#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloConstantPadForwardRunHost(miopenTensorDescriptor_t inputDesc,
                                     miopenTensorDescriptor_t outputDesc,
                                     Tgpu* input,
                                     Tcheck* output,
                                     std::vector<int64_t> padding_vec,
                                     const Tgpu value)
{
    auto input_tv        = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto output_tv       = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    int64_t padding_size = padding_vec.size();

    // Prepare padding
    int64_t padding_args[10] = {0};
    int io_dim_size          = miopen::deref(inputDesc).GetNumDims();
    for(uint64_t i = 0; i < padding_size / 2; i++)
    {
        uint64_t idx              = io_dim_size - i - 1;
        padding_args[idx * 2]     = padding_vec[i * 2];
        padding_args[idx * 2 + 1] = padding_vec[i * 2 + 1];
    }

    size_t output_size = miopen::deref(outputDesc).GetElementSize();
    for(uint64_t gid = 0; gid < output_size; ++gid)
    {
        bool flag            = true;
        auto i_tensor_layout = tensor_layout_t<5>({0, 0, 0, 0, 0});
        auto o_tensor_layout = tensor_layout_t<5>(output_tv, gid);

        for(uint64_t i = 0; i < 5; i++)
        {
            int64_t idx = o_tensor_layout.layout[i] - padding_args[2 * i];
            if(idx < 0 || idx >= input_tv.size[i])
            {
                flag = false;
                break;
            }

            i_tensor_layout.layout[i] = idx;
        }

        output[output_tv.get_tensor_view_idx(o_tensor_layout)] =
            flag ? input[input_tv.get_tensor_view_idx(i_tensor_layout)] : value;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloConstantPadBackwardRunHost(miopenTensorDescriptor_t inputGradDesc,
                                      miopenTensorDescriptor_t outputGradDesc,
                                      Tcheck* input_grad,
                                      Tgpu* output_grad,
                                      std::vector<int64_t>& padding_vec)
{
    auto input_grad_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    auto output_grad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    int io_dim_size     = miopen::deref(inputGradDesc).GetNumDims();

    // Prepare padding
    int64_t padding_args[10] = {0};
    auto padding_size        = padding_vec.size();
    for(uint64_t i = 0; i < padding_size / 2; i++)
    {
        size_t idx                = io_dim_size - i - 1;
        padding_args[idx * 2]     = padding_vec[i * 2];
        padding_args[idx * 2 + 1] = padding_vec[i * 2 + 1];
    }

    auto input_grad_numels = miopen::deref(inputGradDesc).GetElementSize();
    for(size_t gid = 0; gid < input_grad_numels; ++gid)
    {
        bool flag             = true;
        auto ig_tensor_layout = tensor_layout_t<5>(input_grad_tv, gid);
        auto og_tensor_layout = tensor_layout_t<5>({0, 0, 0, 0, 0});

        for(uint64_t i = 0; i < 5; i++)
        {
            int64_t idx = ig_tensor_layout.layout[i] + padding_args[2 * i];
            if(idx < 0 || idx >= output_grad_tv.size[i])
            {
                flag = false;
                break;
            }

            og_tensor_layout.layout[i] = idx;
        }

        input_grad[input_grad_tv.get_tensor_view_idx(ig_tensor_layout)] =
            flag ? output_grad[output_grad_tv.get_tensor_view_idx(og_tensor_layout)]
                 : static_cast<Tcheck>(0);
    }

    return miopenStatusSuccess;
}
