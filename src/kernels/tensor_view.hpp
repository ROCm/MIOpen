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

#ifndef GUARD_TENSOR_VIEW_H
#define GUARD_TENSOR_VIEW_H

template <int N>
struct tensor_layerout_t;

template <int N>
struct tensor_view_t
{
    // Get index in tensor view at tensor layout
    constexpr uint64_t get_tensor_view_idx(tensor_layerout_t<N> tensor_layout)
    {
        uint64_t idx = 0;
        for(auto i = 0; i < N; ++i)
        {
            idx += stride[i] * tensor_layout.layerout[i];
        }
        return idx;
    }
    uint64_t stride[N];
    uint64_t size[N];
};

template <int N>
struct tensor_layerout_t
{
    // Make tensor layout at index using tensor view
    constexpr tensor_layerout_t(tensor_view_t<N>& tensor_view, uint64_t idx)
    {
        uint64_t temp = idx;
        if(N == 1)
        {
            layerout[0] = idx;
        }
        else
        {
            for(auto i = N - 1; i >= 1; --i)
            {
                if(i > 1)
                {
                    layerout[i] = (temp) % tensor_view.size[i];
                }
                else
                {
                    layerout[i - 1] = temp / tensor_view.size[i];
                    layerout[i]     = temp % tensor_view.size[i];
                }
                temp = idx / tensor_view.size[i];
            }
        }
    }

    uint64_t layerout[N];
};

#endif // GUARD_TENSOR_VIEW_H
