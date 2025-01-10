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

#include <cmath>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck, typename Ta>
int32_t mloFractionalMaxPoolForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                           const Tgpu* input,
                                           const miopenTensorDescriptor_t targetDesc,
                                           const Ta* target,
                                           const miopenTensorDescriptor_t outputDesc,
                                           Tcheck* output,
                                           const miopenTensorDescriptor_t backpropDesc,
                                           Tcheck* backprop,
                                           const uint64_t num_class)
{
    auto input_tv    = miopen::get_inner_expanded_tv<2>(miopen::deref(inputDesc));
    auto target_tv   = miopen::get_inner_expanded_tv<1>(miopen::deref(targetDesc));
    auto output_tv   = miopen::get_inner_expanded_tv<1>(miopen::deref(outputDesc));
    auto backprop_tv = miopen::get_inner_expanded_tv<2>(miopen::deref(backpropDesc));

    par_ford(miopen::deref(inputDesc).GetLengths()[0])([&](auto gid) {
        double lmax    = std::numeric_limits<double>::lowest();
        double lsum    = 0.0f;
        uint64_t label = static_cast<uint64_t>(target[target_tv.get_tensor_view_idx({gid})]);

        ford(num_class)([&](uint64_t j) {
            double val = static_cast<double>(input[input_tv.get_tensor_view_idx({gid, j})]);
            lmax       = std::max(lmax, val);
        });

        ford(num_class)([&](uint64_t j) {
            double val = static_cast<double>(input[input_tv.get_tensor_view_idx({gid, j})]);
            lsum += std::exp(val - lmax);
        });

        double val = static_cast<double>(input[input_tv.get_tensor_view_idx({gid, label})]);
        output[output_tv.get_tensor_view_idx({gid})] =
            static_cast<Tcheck>(std::log(lsum) - val + lmax);

        par_ford(num_class)([&](uint64_t j) {
            double val = static_cast<double>(input[input_tv.get_tensor_view_idx({gid, j})]);
            double backprop_val =
                (j == label) ? std::exp(val - lmax) / lsum - 1.0f : std::exp(val - lmax) / lsum;

            backprop[backprop_tv.get_tensor_view_idx({gid, j})] = static_cast<Tcheck>(backprop_val);
        });
    });

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tcheck>
int32_t mloFractionalMaxPoolBackwardRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                            const Tgpu* output_grad,
                                            const miopenTensorDescriptor_t backpropDesc,
                                            const Tgpu* backprop,
                                            const miopenTensorDescriptor_t inputGradDesc,
                                            Tcheck* input_grad,
                                            const uint64_t num_class)
{
    auto output_grad_tv = miopen::get_inner_expanded_tv<1>(miopen::deref(outputGradDesc));
    auto backprop_tv    = miopen::get_inner_expanded_tv<2>(miopen::deref(backpropDesc));
    auto input_grad_tv  = miopen::get_inner_expanded_tv<2>(miopen::deref(inputGradDesc));

    par_ford(miopen::deref(outputGradDesc).GetLengths()[0])([&](auto gid) {
        double output_grad_val =
            static_cast<double>(output_grad[output_grad_tv.get_tensor_view_idx({gid})]);

        par_ford(num_class)([&](uint64_t j) {
            double backprop_val =
                static_cast<double>(backprop[backprop_tv.get_tensor_view_idx({gid, j})]);

            input_grad[input_grad_tv.get_tensor_view_idx({gid, j})] =
                static_cast<Tcheck>(output_grad_val * backprop_val);
        });
    });
    return miopenStatusSuccess;
}
