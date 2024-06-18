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

#ifndef MIOPEN_TENSOR_VIEW_UTIL_HPP_
#define MIOPEN_TENSOR_VIEW_UTIL_HPP_

#include <cstdlib>
#include <miopen/common.hpp>
#include "../../kernels/tensor_view.hpp"
#include "miopen/tensor.hpp"

namespace miopen {

inline tensor_view_t<5> get_inner_expanded_5dtv(const TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_t<5> tensor_view;
    for(size_t i = 0; i < 5; ++i)
    {
        if(i < dims.size())
        {
            tensor_view.stride[i] = strides[i];
            tensor_view.size[i]   = dims[i];
        }
        else
        {
            tensor_view.stride[i] = (i == 0 ? 1 : tensor_view.stride[i - 1]);
            tensor_view.size[i]   = 1;
        }
    }
    return tensor_view;
}

template <int N>
inline bool isTensorViewContiguous(const tensor_view_t<N>& tv)
{
    size_t planeSize = 1;
    for(int i = N - 1; i >= 0; i--)
    {
        if(tv.stride[i] != planeSize)
            return false;
        planeSize *= tv.size[i];
    }
    return true;
}

} // namespace miopen

#endif // MIOPEN_TENSOR_REORDER_UTIL_HPP_
