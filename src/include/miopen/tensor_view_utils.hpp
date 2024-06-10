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

#include <miopen/common.hpp>
#include "../../kernels/tensor_view.hpp"
#include "miopen/tensor.hpp"

namespace miopen {

template <int N>
inline tensor_view_t<N> get_inner_expanded_tv(const TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_t<N> tensor_view;
    for(size_t i = 0; i < N; ++i)
    {
        if(i < dims.size())
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

inline bool isBroadcastable(const TensorDescriptor x, const TensorDescriptor y)
{
    auto xLen = x.GetLengths();
    auto yLen = y.GetLengths();
    if(xLen.empty() || yLen.empty())
        return false;

    int trailIdxX = xLen.size() - 1;
    int trailIdxY = yLen.size() - 1;
    while(trailIdxX >= 0 && trailIdxY >= 0)
    {
        if(xLen[trailIdxX] != yLen[trailIdxY] && xLen[trailIdxX] != 1 && yLen[trailIdxY] != 1)
            return false;
        trailIdxX--;
        trailIdxY--;
    }
    return true;
}

template <int N>
inline tensor_view_t<N> broadcast_to(const tensor_view_t<N> in, const tensor_view_t<N> target)
{
    tensor_view_t<N> out;
    for(int i = 0; i < N; i++)
    {
        if(in.size[i] == target.size[i])
        {
            out.size[i]   = in.size[i];
            out.stride[i] = in.stride[i];
        }
        else
        {
            out.size[i]   = target.size[i];
            out.stride[i] = 0;
        }
    }
    return out;
}

} // namespace miopen

#endif // MIOPEN_TENSOR_REORDER_UTIL_HPP_
