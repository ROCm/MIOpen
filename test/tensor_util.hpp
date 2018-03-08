/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#ifndef GUARD_TENSOR_UTIL_HPP
#define GUARD_TENSOR_UTIL_HPP

#include <iostream>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <utility>
#include <cstdlib>
#include "tensor_holder.hpp"

// loop over sub-tensor, and operate on each data
template <typename T, template <typename> class data_operator_t>
void operate_over_subtensor(const data_operator_t<T>& r_data_operator,
                            tensor<T>& rSuperTensor,
                            const miopen::TensorDescriptor& rSubDesc,
                            const int offset)
{
    operate_over_subtensor_impl(r_data_operator, rSuperTensor, rSubDesc, 0, offset);
}

// loop over part of sub-tensor (dimensions lower than "current_dim"), and operate on
// each data
template <typename T, template <typename> class data_operator_t>
void operate_over_subtensor_impl(const data_operator_t<T>& r_data_operator,
                                 tensor<T>& rSuperTensor,
                                 const miopen::TensorDescriptor& rSubDesc,
                                 const uint current_dim,
                                 const int offset)
{
    const int max_dim        = rSubDesc.GetLengths().size() - 1;
    const int current_stride = rSubDesc.GetStrides()[current_dim];

    int index = offset;

    for(int i = 0; i < rSubDesc.GetLengths()[current_dim]; ++i)
    {
        if(current_dim == max_dim)
            r_data_operator(rSuperTensor[index]);
        else
            operate_over_subtensor_impl<T, data_operator_t>(
                r_data_operator, rSuperTensor, rSubDesc, current_dim + 1, index);

        index += current_stride;
    }
}

#endif
