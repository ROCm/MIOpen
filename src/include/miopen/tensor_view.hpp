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

#include "miopen/tensor.hpp"

using tensor_view_1d_t = struct
{
    uint64_t stride[1];
    uint64_t size[1];
};

using tensor_view_2d_t = struct
{
    uint64_t stride[2];
    uint64_t size[2];
};

using tensor_view_3d_t = struct
{
    uint64_t stride[3];
    uint64_t size[3];
};

using tensor_view_4d_t = struct
{
    uint64_t stride[4];
    uint64_t size[4];
};

using tensor_view_5d_t = struct
{
    uint64_t stride[5];
    uint64_t size[5];
};

#define TV_IDX(tv, d, n) (tv.stride[d] * (n))

#define TV1D_IDX(tv, n0) (TV_IDX(tv, 0, n0))

#define TV2D_IDX(tv, n0, n1) (TV_IDX(tv, 1, n1) + TV1D_IDX(tv, n0))

#define TV3D_IDX(tv, n0, n1, n2) (TV_IDX(tv, 2, n2) + TV2D_IDX(tv, n0, n1))

#define TV4D_IDX(tv, n0, n1, n2, n3) (TV_IDX(tv, 3, n3) + TV3D_IDX(tv, n0, n1, n2))

#define TV5D_IDX(tv, n0, n1, n2, n3, n4) (TV_IDX(tv, 4, n4) + TV4D_IDX(tv, n0, n1, n2, n3))

#define TV_1D_AT(x, idx) (x[IDX_TO_TV1D_IDX(x##_tv, idx)])
#define TV_2D_AT(x, n0, n1) (x[TV2D_IDX(x##_tv, n0, n1)])
#define TV_3D_AT(x, n0, n1, n2) (x[TV3D_IDX(x##_tv, n0, n1, n2)])
#define TV_4D_AT(x, n0, n1, n2, n3) (x[TV4D_IDX(x##_tv, n0, n1, n2, n3)])
#define TV_5D_AT(x, n0, n1, n2, n3, n4) (x[TV5D_IDX(x##_tv, n0, n1, n2, n3, n4)])

#define GET_NCDHW(n, c, d, h, w, idx, tv) \
    {                                     \
        ulong ncdh = (idx) / tv.size[4];  \
        w          = (idx) % tv.size[4];  \
        ulong ncd  = ncdh / tv.size[3];   \
        h          = ncdh % tv.size[3];   \
        ulong nc   = ncd / tv.size[2];    \
        d          = ncd % tv.size[2];    \
        n          = nc / tv.size[1];     \
        c          = nc % tv.size[1];     \
    }

#define GET_NCDH(n, c, d, h, idx, tv)   \
    {                                   \
        ulong ncd = (idx) / tv.size[3]; \
        h         = (idx) % tv.size[3]; \
        ulong nc  = ncd / tv.size[2];   \
        d         = ncd % tv.size[2];   \
        n         = nc / tv.size[1];    \
        c         = nc % tv.size[1];    \
    }

#define GET_NCD(n, c, d, idx, tv)      \
    {                                  \
        ulong nc = (idx) / tv.size[2]; \
        d        = (idx) % tv.size[2]; \
        n        = nc / tv.size[1];    \
        c        = nc % tv.size[1];    \
    }

inline tensor_view_1d_t get_inner_expanded_tv_1d(const miopen::TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_1d_t tv_1d;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        tv_1d.stride[i] = strides[i];
        tv_1d.size[i]   = dims[i];
    }
    auto rest = strides.size();
    for(size_t j = rest; j < 1; ++j)
    {
        tv_1d.stride[j] = (rest == 0 ? 1 : strides[rest - 1]);
        tv_1d.size[j]   = 1;
    }
    return tv_1d;
}

inline tensor_view_2d_t get_inner_expanded_tv_2d(const miopen::TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_2d_t tv_2d;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        tv_2d.stride[i] = strides[i];
        tv_2d.size[i]   = dims[i];
    }
    auto rest = strides.size();
    for(size_t j = rest; j < 2; ++j)
    {
        tv_2d.stride[j] = (rest == 0 ? 1 : strides[rest - 1]);
        tv_2d.size[j]   = 1;
    }
    return tv_2d;
}

inline tensor_view_3d_t get_inner_expanded_tv_3d(const miopen::TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_3d_t tv_3d;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        tv_3d.stride[i] = strides[i];
        tv_3d.size[i]   = dims[i];
    }
    auto rest = strides.size();
    for(size_t j = rest; j < 3; ++j)
    {
        tv_3d.stride[j] = (rest == 0 ? 1 : strides[rest - 1]);
        tv_3d.size[j]   = 1;
    }
    return tv_3d;
}

inline tensor_view_4d_t get_inner_expanded_tv_4d(const miopen::TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_4d_t tv_4d;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        tv_4d.stride[i] = strides[i];
        tv_4d.size[i]   = dims[i];
    }
    auto rest = strides.size();
    for(size_t j = rest; j < 4; ++j)
    {
        tv_4d.stride[j] = (rest == 0 ? 1 : strides[rest - 1]);
        tv_4d.size[j]   = 1;
    }
    return tv_4d;
}

inline tensor_view_5d_t get_inner_expanded_tv_5d(const miopen::TensorDescriptor Desc)
{
    auto dims    = Desc.GetLengths();
    auto strides = Desc.GetStrides();

    tensor_view_5d_t tv_5d;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        tv_5d.stride[i] = strides[i];
        tv_5d.size[i]   = dims[i];
    }
    auto rest = strides.size();
    for(size_t j = rest; j < 5; ++j)
    {
        tv_5d.stride[j] = (rest == 0 ? 1 : strides[rest - 1]);
        tv_5d.size[j]   = 1;
    }
    return tv_5d;
}

#endif // GUARD_TENSOR_VIEW_H
