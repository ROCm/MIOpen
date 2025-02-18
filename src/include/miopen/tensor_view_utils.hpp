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

#include "../../kernels/tensor_view.hpp"
#include <miopen/tensor.hpp>

namespace miopen {

template <int N>
inline tensor_view_t<N> get_inner_expanded_tv(const TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_t<N> tensor_view{};
    for(size_t i = 0; i < N; ++i)
    {
        if(dims.empty())
        {
            tensor_view.stride[i] = 0;
            tensor_view.size[i]   = 0;
        }
        else if(i < dims.size())
        {
            tensor_view.stride[i] = strides[i];
            tensor_view.size[i]   = dims[i];
        }
        else
        {
            tensor_view.stride[i] = (i == 0 ? 1 : strides[i - 1]);
            tensor_view.size[i]   = 1;
        }
    }
    return tensor_view;
}

template <int N>
inline void slice_tv(tensor_view_t<N>& tensor_view, int32_t sliceCount, const int32_t* slices)
{
    for(int32_t i = 0; i < sliceCount; i++)
    {
        int32_t dim   = slices[4 * i + 0];
        int32_t start = slices[4 * i + 1];
        int32_t end   = slices[4 * i + 2];
        int32_t step  = slices[4 * i + 3];

        if(end > static_cast<int32_t>(tensor_view.size[dim]))
            end = tensor_view.size[dim];

        auto len = end - start;

        tensor_view.size[dim] = (len + step - 1) / step;
        tensor_view.stride[dim] *= step;
    }
}

template <int N>
inline tensor_view_t<N - 1> get_tv_without_dim(const tensor_view_t<N>& origin_tv, int selected_dim)
{
    tensor_view_t<N - 1> res{};
    for(int i = 0; i < N; ++i)
    {
        if(i == selected_dim)
            continue;
        if(i < selected_dim)
        {
            res.size[i]   = origin_tv.size[i];
            res.stride[i] = origin_tv.stride[i];
        }
        else
        {
            res.size[i - 1]   = origin_tv.size[i];
            res.stride[i - 1] = origin_tv.stride[i];
        }
    }
    return res;
}

} // namespace miopen

#endif // MIOPEN_TENSOR_VIEW_UTIL_HPP_
