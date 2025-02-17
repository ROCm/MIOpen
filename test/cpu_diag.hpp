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

#include "ford.hpp"
#include "tensor_holder.hpp"
#include "tensor_view.hpp"

#include <miopen/diag/solvers.hpp>
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_diag_forward(const tensor<T>& input, tensor<T>& ref_output, int64_t diagonal)
{
    if(input.desc.GetNumDims() == 1)
    {
        auto input_numel = input.desc.GetElementSize();
        auto output_tv   = miopen::get_inner_expanded_tv<2>(ref_output.desc);
        auto offset =
            (diagonal >= 0) ? diagonal * output_tv.stride[1] : -diagonal * output_tv.stride[0];

        par_ford(input_numel)([&](size_t o) {
            long outputIdx        = o * (output_tv.stride[0] + output_tv.stride[1]) + offset;
            ref_output[outputIdx] = input[o];
        });
    }
    else if(input.desc.GetNumDims() == 2)
    {
        auto output_numel = ref_output.desc.GetElementSize();
        auto input_tv     = miopen::get_inner_expanded_tv<2>(input.desc);
        auto output_tv    = miopen::get_inner_expanded_tv<1>(ref_output.desc);
        auto offset =
            (diagonal >= 0) ? diagonal * input_tv.stride[1] : -diagonal * input_tv.stride[0];

        par_ford(output_numel)([&](size_t o) {
            long inputIdx = o * (input_tv.stride[0] + input_tv.stride[1]) + offset;
            setNDVal(ref_output.data.data(),
                     output_tv,
                     o,
                     getNDVal(input.data.data(), input_tv, inputIdx));
        });
    }
}
