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
#include "tensor_view.hpp"
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_allclose_forward(const tensor<T> input1,
                          const tensor<T> input2,
                          tensor<int32_t>& output,
                          const float atol,
                          const float rtol,
                          const bool equal_nan)
{
    auto input1_tv = miopen::get_inner_expanded_tv<5>(input1.desc);
    auto input2_tv = miopen::get_inner_expanded_tv<5>(input2.desc);

    output[0] = 1;

    for(uint64_t gid = 0; gid < input1.desc.GetElementSize(); gid++)
    {
        tensor_layout_t<5> layout(input1_tv, gid);

        double input1_fvalue = static_cast<double>(input1[input1_tv.get_tensor_view_idx(layout)]);
        double input2_fvalue = static_cast<double>(input2[input2_tv.get_tensor_view_idx(layout)]);

        int32_t out;
        bool input1_isnan = std::isnan(input1_fvalue);
        bool input2_isnan = std::isnan(input2_fvalue);

        if(input1_isnan || input2_isnan)
        {
            if(equal_nan == 1 && input1_isnan == true && input2_isnan == true)
            {
                out = 1;
            }
            else
            {
                out = 0;
            }
        }
        else
        {
            if(std::fabs(input1_fvalue - input2_fvalue) <= (atol + rtol * std::fabs(input2_fvalue)))
            {
                out = 1;
            }
            else
            {
                out = 0;
            }
        }
        if(out == 0)
        {
            output[0] = 0;
            break;
        }
    }
}
