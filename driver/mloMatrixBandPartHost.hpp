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
#include <cstdint>
#include <miopen/tensor.hpp>
#include <miopen/tensor_view_utils.hpp>
#include <../test/ford.hpp>

template <typename Tgpu, typename Tcheck, typename Tn>
int32_t mloMatrixBandPartRunHost(const miopenTensorDescriptor_t inputDesc,
                                 const Tgpu* input,
                                 const miopenTensorDescriptor_t outputDesc,
                                 Tcheck* output,
                                 const Tn* num_lower,
                                 const Tn* num_upper)
{
    auto input_tv    = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto output_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));
    uint64_t num_dim = miopen::deref(inputDesc).GetNumDims();

    par_ford(miopen::deref(inputDesc).GetElementSize())([&](uint64_t gid) {
        int64_t w = gid % input_tv.size[num_dim - 1];
        int64_t h = (gid / input_tv.size[num_dim - 1]) % input_tv.size[num_dim - 2];

        int64_t num_lower_val = static_cast<int64_t>(num_lower[0]);
        int64_t num_upper_val = static_cast<int64_t>(num_upper[0]);
        int64_t diff          = h - w;

        bool in_band = (num_lower_val < 0 || diff <= num_lower_val) &&
                       (num_upper_val < 0 || (-diff) <= num_upper_val);

        tensor_layout_t<5> layout(input_tv, gid);

        output[output_tv.get_tensor_view_idx(layout)] =
            in_band ? static_cast<Tcheck>(input[input_tv.get_tensor_view_idx(layout)])
                    : static_cast<Tcheck>(0);
    });

    return miopenStatusSuccess;
}
