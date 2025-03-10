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

#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

template <class T, int NDIM>
void cpu_mseloss_forward(const tensor<T> input,
                         const tensor<T> target,
                         tensor<T>& ref_output,
                         const miopenLossReductionMode_t reduction)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv<NDIM>(input.desc);
    auto T_tv = get_inner_expanded_tv<NDIM>(target.desc);
    auto O_tv = get_inner_expanded_tv<NDIM>(ref_output.desc);

    auto size = input.desc.GetElementSize();

    std::vector<double> buffer;
    if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        buffer.assign(size, 0);

    par_ford(size)([&](size_t i) {
        const auto tensor_layout = tensor_layout_t<5>(I_tv, i);
        const uint64_t Iidx      = I_tv.get_tensor_view_idx(tensor_layout);
        const uint64_t Tidx      = T_tv.get_tensor_view_idx(tensor_layout);
        float sub  = static_cast<float>(input[Iidx]) - static_cast<float>(target[Tidx]);
        float loss = sub * sub;
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
            ref_output[O_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(loss);
        else
            buffer[i] = loss;
    });

    double loss_sum = std::accumulate(buffer.begin(), buffer.end(), 0.0);

    if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        loss_sum /= size;
    if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        ref_output[0] = static_cast<T>(loss_sum);
}

template <class T, int NDIM>
void cpu_mseloss_backward(tensor<T> input,
                          tensor<T> target,
                          tensor<T> dO,
                          tensor<T>& ref_dI,
                          tensor<T>& ref_dT,
                          miopenLossReductionMode_t reduction)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv  = get_inner_expanded_tv<5>(input.desc);
    auto T_tv  = get_inner_expanded_tv<5>(target.desc);
    auto dO_tv = get_inner_expanded_tv<5>(dO.desc);
    auto dI_tv = get_inner_expanded_tv<5>(ref_dI.desc);
    auto dT_tv = get_inner_expanded_tv<5>(ref_dT.desc);

    auto size = input.desc.GetElementSize();

    par_ford(size)([&](size_t i) {
        const auto tensor_layout = tensor_layout_t<5>(I_tv, i);
        const uint64_t Iidx      = I_tv.get_tensor_view_idx(tensor_layout);
        const uint64_t Tidx      = T_tv.get_tensor_view_idx(tensor_layout);

        float sub  = static_cast<float>(input[Iidx]) - static_cast<float>(target[Tidx]);
        float grad = 2.0f * sub *
                     static_cast<float>(dO[reduction == MIOPEN_LOSS_REDUCTION_NONE
                                               ? dO_tv.get_tensor_view_idx(tensor_layout)
                                               : 0]);

        if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
            grad = grad / size;

        ref_dI[dI_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(grad);
        ref_dT[dT_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(-grad);
    });
}
