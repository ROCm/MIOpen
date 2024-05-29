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

#ifndef GUARD_TENSOR_VIEW_HPP
#define GUARD_TENSOR_VIEW_HPP

template <int N>
struct tensor_layout_t;

template <int N>
struct tensor_view_t
{
    // Get index in tensor view at tensor layout
    constexpr uint64_t get_tensor_view_idx(const tensor_layout_t<N>& tensor_layout)
    {
        static_assert(N > 0);
        uint64_t idx = 0;
        for(auto i = 0; i < N; ++i)
        {
            idx += stride[i] * tensor_layout.layout[i];
        }
        return idx;
    }
    uint64_t stride[N];
    uint64_t size[N];
};

template <int N>
struct tensor_layout_t
{
    // Make tensor layout at index using tensor view
    constexpr tensor_layout_t(const tensor_view_t<N>& tensor_view, uint64_t idx)
    {
        static_assert(N > 0);
        uint64_t temp = idx;
        if constexpr(N == 1)
        {
            layout[0] = idx;
        }
        else
        {
            for(auto i = N - 1; i > 1; --i)
            {
                layout[i] = temp % tensor_view.size[i];
                temp      = temp / tensor_view.size[i];
            }
            layout[1] = temp % tensor_view.size[1];
            layout[0] = temp / tensor_view.size[1];
        }
    }

    uint64_t layout[N];
};

#endif // GUARD_TENSOR_VIEW_HPP
