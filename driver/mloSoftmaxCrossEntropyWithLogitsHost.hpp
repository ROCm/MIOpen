/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
int32_t mloSoftmaxCrossEntropyWithLogitsForward(const miopenTensorDescriptor_t inputDesc,
                                                const miopenTensorDescriptor_t targetDesc,
                                                const miopenTensorDescriptor_t outputDesc,
                                                const miopenTensorDescriptor_t backpropDesc,
                                                const Tgpu* input,
                                                const Tgpu* target,
                                                Tcheck* output,
                                                Tcheck* backprop)
{
    auto I_tv = get_inner_expanded_tv<2>(miopen::deref(inputDesc));
    auto T_tv = get_inner_expanded_tv<2>(miopen::deref(targetDesc));
    auto O_tv = get_inner_expanded_tv<1>(miopen::deref(outputDesc));
    auto B_tv = get_inner_expanded_tv<2>(miopen::deref(backpropDesc));

    size_t num_batches = miopen::deref(inputDesc).GetLengths()[0];
    size_t num_class   = miopen::deref(inputDesc).GetLengths()[1];

    for(size_t gid = 0; gid < num_batches; ++gid)
    {
        float max_val = -std::numeric_limits<float>::infinity();

        for(size_t i = 0; i < num_class; ++i)
        {
            size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
            float val   = static_cast<float>(input[Iidx]);
            max_val     = std::max(max_val, val);
        }

        float sum = 0.0f;
        for(size_t i = 0; i < num_class; ++i)
        {
            size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
            sum += std::exp(static_cast<float>(input[Iidx]) - max_val);
        }

        float log_sum = std::log(sum);
        float loss    = 0.0f;
        for(size_t i = 0; i < num_class; ++i)
        {
            size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
            size_t Tidx = T_tv.get_tensor_view_idx({gid, i});
            float val   = static_cast<float>(input[Iidx]);
            float label = static_cast<float>(target[Tidx]);
            loss += label * (log_sum - val + max_val);
        }

        size_t Oidx  = O_tv.get_tensor_view_idx({gid});
        output[Oidx] = static_cast<Tcheck>(loss);

        for(size_t i = 0; i < num_class; ++i)
        {
            size_t Iidx        = I_tv.get_tensor_view_idx({gid, i});
            size_t Tidx        = T_tv.get_tensor_view_idx({gid, i});
            size_t Bidx        = B_tv.get_tensor_view_idx({gid, i});
            float val          = static_cast<float>(input[Iidx]);
            float label        = static_cast<float>(target[Tidx]);
            float backprop_val = std::exp(val - max_val) / sum - label;
            backprop[Bidx]     = static_cast<Tcheck>(backprop_val);
        }
    }
    return 0;
}

template <typename Tgpu, typename Tcheck>
int32_t mloSoftmaxCrossEntropyWithLogitsBackward(const miopenTensorDescriptor_t outputGradDesc,
                                                 const miopenTensorDescriptor_t backpropDesc,
                                                 const miopenTensorDescriptor_t inputDesc,
                                                 const miopenTensorDescriptor_t inputGradDesc,
                                                 const miopenTensorDescriptor_t targetGradDesc,
                                                 const Tgpu* output_grad,
                                                 const Tgpu* backprop,
                                                 const Tgpu* input,
                                                 Tcheck* input_grad,
                                                 Tcheck* target_grad,
                                                 bool input_grad_out,
                                                 bool target_grad_out)
{
    auto dO_tv = get_inner_expanded_tv<1>(miopen::deref(outputGradDesc));
    auto B_tv  = get_inner_expanded_tv<2>(miopen::deref(backpropDesc));
    auto I_tv  = get_inner_expanded_tv<2>(miopen::deref(inputDesc));

    size_t num_batches = miopen::deref(inputDesc).GetLengths()[0];
    size_t num_class   = miopen::deref(inputDesc).GetLengths()[1];

    for(size_t gid = 0; gid < num_batches; ++gid)
    {
        size_t dOidx          = dO_tv.get_tensor_view_idx({gid});
        float output_grad_val = static_cast<float>(output_grad[dOidx]);

        if(input_grad_out)
        {
            auto dI_tv = get_inner_expanded_tv<2>(miopen::deref(inputGradDesc));
            for(size_t i = 0; i < num_class; ++i)
            {
                size_t Bidx        = B_tv.get_tensor_view_idx({gid, i});
                size_t dIidx       = dI_tv.get_tensor_view_idx({gid, i});
                float backprop_val = static_cast<float>(backprop[Bidx]);
                input_grad[dIidx]  = static_cast<Tcheck>(output_grad_val * backprop_val);
            }
        }

        if(target_grad_out)
        {
            auto dT_tv    = get_inner_expanded_tv<2>(miopen::deref(targetGradDesc));
            float max_val = -std::numeric_limits<float>::infinity();
            for(size_t i = 0; i < num_class; ++i)
            {
                size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
                float val   = static_cast<float>(input[Iidx]);
                max_val     = std::max(max_val, val);
            }

            float sum = 0.0f;
            for(size_t i = 0; i < num_class; ++i)
            {
                size_t Iidx = I_tv.get_tensor_view_idx({gid, i});
                float val   = static_cast<float>(input[Iidx]);
                sum += std::exp(val - max_val);
            }

            float log_val = std::log(sum);
            for(size_t i = 0; i < num_class; ++i)
            {
                size_t Iidx     = I_tv.get_tensor_view_idx({gid, i});
                float logit_val = static_cast<float>(input[Iidx]);
                size_t Tidx     = dT_tv.get_tensor_view_idx({gid, i});
                target_grad[Tidx] =
                    static_cast<Tcheck>((max_val + log_val - logit_val) * output_grad_val);
            }
        }
    }
    return 0;
}
