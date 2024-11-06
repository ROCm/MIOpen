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

#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck>
int32_t mloCartesianProdForwardRunHost(std::vector<miopenTensorDescriptor_t> inputDescs,
                                       const miopenTensorDescriptor_t outputDesc,
                                       const std::vector<Tgpu*> inputs,
                                       Tcheck* output)
{
    auto output_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(outputDesc));

    Tcheck* output_ws = new Tcheck[output_tv.size[0] * output_tv.size[1]];
    size_t stride     = 1;
    par_ford(inputs.size())([&](size_t dim1_idx) 
    {
        dim1_idx = inputs.size() - 1 - dim1_idx;
        auto input_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(inputDescs[dim1_idx]));
        auto numel    = miopen::deref(outputDesc).GetLengths()[0];

        par_ford(numel)([&](size_t gid) {
            if(gid >= output_tv.size[0])
                return;

            output_ws[output_tv.size[0] * dim1_idx + gid] =
                inputs[dim1_idx][(gid / stride) % input_tv.size[0]];
        });
        stride *= miopen::deref(inputDescs[dim1_idx]).GetElementSize();
    });

    par_ford(output_tv.size[1])([&](size_t dim1_idx) {
        par_ford(output_tv.size[0])([&](size_t dim0_idx) {
            output[output_tv.get_tensor_view_idx({dim0_idx, dim1_idx})] =
                output_ws[output_tv.size[0] * dim1_idx + dim0_idx];
        });
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloCartesianProdBackwardRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                        std::vector<miopenTensorDescriptor_t> inputGradDescs,
                                        const Tgpu* output_grad,
                                        std::vector<Tcheck*>& input_grads)
{
    auto output_grad_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(outputGradDesc));
    size_t stride       = 1;
    par_ford(input_grads.size())([&](size_t dim1_idx) 
    {
        dim1_idx = input_grads.size() - 1 - dim1_idx;
        auto input_grad_tv =
            miopen::get_inner_expanded_tv<1>(miopen::deref(inputGradDescs[dim1_idx]));
        auto numel = miopen::deref(inputGradDescs[dim1_idx]).GetElementSize();

        par_ford(numel)([&](size_t gid) {
            if(gid >= input_grad_tv.size[0])
                return;
            float sum = 0;
            for(size_t offset = 0; offset < output_grad_tv.size[0];
                offset += (stride * input_grad_tv.size[0]))
            {
                for(size_t i = 0; i < stride; ++i)
                {
                    size_t dim0_idx = offset + gid * stride + i;
                    sum += static_cast<float>(
                        output_grad[output_grad_tv.get_tensor_view_idx({dim0_idx, dim1_idx})]);
                }
            }

            input_grads[dim1_idx][input_grad_tv.get_tensor_view_idx({gid})] =
                static_cast<Tcheck>(sum);
        });
        stride *= numel;
    });
    return miopenStatusSuccess;
}
