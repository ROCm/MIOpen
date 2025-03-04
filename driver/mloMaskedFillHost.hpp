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

#include <miopen/errors.hpp>
#include <miopen/tensor_view_utils.hpp>

template <typename Tgpu, typename Tcheck>
int mloMaskedFillForwardRunHost(const miopenTensorDescriptor_t inputDesc,
                                const miopenTensorDescriptor_t maskDesc,
                                const miopenTensorDescriptor_t outputDesc,
                                const Tgpu* input,
                                Tcheck* outputHost,
                                const int8_t* mask,
                                float value)
{
    auto numel                 = miopen::deref(outputDesc).GetElementSize();
    tensor_view_t<5> output_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    tensor_view_t<5> input_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    tensor_view_t<5> mask_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(maskDesc));

    for(auto i = 0; i < numel; ++i)
    {
        tensor_layout_t<5> output_layout{output_tv, i};
        tensor_layout_t<5> input_layout{input_tv, i};
        tensor_layout_t<5> mask_layout{mask_tv, i};
        outputHost[output_tv.get_tensor_view_idx(output_layout)] =
            mask[mask_tv.get_tensor_view_idx(mask_layout)]
                ? value
                : input[input_tv.get_tensor_view_idx(input_layout)];
    }

    return 0;
}

template <typename Tgpu, typename Tcheck>
int mloMaskedFillBackwardRunHost(const miopenTensorDescriptor_t outputGradDesc,
                                 const miopenTensorDescriptor_t maskDesc,
                                 const miopenTensorDescriptor_t inputGradDesc,
                                 const Tgpu* outputGrad,
                                 Tcheck* inputGrad,
                                 const int8_t* mask)
{
    auto numel = miopen::deref(outputGradDesc).GetElementSize();
    tensor_view_t<5> outputGrad_tv =
        miopen::get_inner_expanded_tv<5>(miopen::deref(outputGradDesc));
    tensor_view_t<5> inputGrad_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(inputGradDesc));
    tensor_view_t<5> mask_tv      = miopen::get_inner_expanded_tv<5>(miopen::deref(maskDesc));

    for(auto i = 0; i < numel; ++i)
    {
        tensor_layout_t<5> outputGrad_layout{outputGrad_tv, i};
        tensor_layout_t<5> inputGrad_layout{inputGrad_tv, i};
        tensor_layout_t<5> mask_layout{mask_tv, i};
        inputGrad[inputGrad_tv.get_tensor_view_idx(inputGrad_layout)] =
            mask[mask_tv.get_tensor_view_idx(mask_layout)]
                ? 0
                : outputGrad[outputGrad_tv.get_tensor_view_idx(outputGrad_layout)];
    }

    return 0;
}
