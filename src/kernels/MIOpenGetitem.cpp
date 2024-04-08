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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "hip_atomic.hpp"
#include "miopen_cstdint.hpp"
#include "float_types.h"
#include "tensor_view.h"

template <typename IDX, typename E>
__device__ void getitembuildindices(const IDX* __restrict__ index,
                                    IDX* __restrict__ element_index,
                                    E* __restrict__ error,
                                    int32_t index_dim,
                                    int32_t indexCount,
                                    int32_t dim_size,
                                    tensor_view_5d_t index_tv,
                                    int32_t dim_offset,
                                    int32_t dim_info_offset)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t NCDHW[5];
    GET_NCDHW(NCDHW[0], NCDHW[1], NCDHW[2], NCDHW[3], NCDHW[4], gid, index_tv);

    if(NCDHW[0] >= index_tv.size[0])
        return;

    uint64_t idx      = TV5D_IDX(index_tv, NCDHW[0], NCDHW[1], NCDHW[2], NCDHW[3], NCDHW[4]);
    IDX getitem_index = index[idx];

    if(getitem_index >= 0 && getitem_index < dim_size)
    {
        element_index[(gid * indexCount) + dim_offset] = getitem_index;
    }
    else if(getitem_index >= -dim_size && getitem_index < 0)
    {
        element_index[(gid * indexCount) + dim_offset] = getitem_index + dim_size;
    }
    else
    {
        error[dim_offset] = -1;
    }

    if(gid == 0)
    {
        element_index[dim_info_offset + dim_offset] = index_dim;
    }
}

template <typename TI, typename IDX, typename TO>
__device__ void getitembwd(const TI* __restrict__ dy,
                           IDX* __restrict__ element_index,
                           TO* __restrict__ dx,
                           int32_t start_dim,
                           int32_t indexCount,
                           tensor_view_5d_t dy_tv,
                           tensor_view_5d_t dx_tv,
                           int32_t dim_info_offset,
                           int32_t offset)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t NCDHW[5];

    GET_NCDHW(NCDHW[0], NCDHW[1], NCDHW[2], NCDHW[3], NCDHW[4], gid, dy_tv);

    if(NCDHW[0] >= dy_tv.size[0])
        return;

    uint64_t idx[5];
    for(uint64_t i = 0; i < 5; ++i)
    {
        idx[i] = NCDHW[i];
    }

    if(indexCount > 0)
    {
        int32_t dim_cursor = NCDHW[start_dim];
        int32_t i          = start_dim;
        int32_t j          = 0;

        for(; i < start_dim + indexCount; ++i, ++j)
        {
            uint64_t dim_idx = static_cast<uint64_t>(element_index[dim_info_offset + j]);
            idx[dim_idx]     = static_cast<uint64_t>(element_index[(dim_cursor * indexCount) + j]);
        }

        i          = element_index[dim_info_offset + indexCount - 1] + 1;
        dim_cursor = start_dim + 1;
        for(; i < 5; ++i, ++dim_cursor)
        {
            idx[i] = NCDHW[dim_cursor];
        }
    }

    atomic_add_g(
        &TV_5D_AT(dx, idx[0] + static_cast<uint64_t>(offset), idx[1], idx[2], idx[3], idx[4]),
        TV_5D_AT(
            dy, NCDHW[0] + static_cast<uint64_t>(offset), NCDHW[1], NCDHW[2], NCDHW[3], NCDHW[4]));
}

extern "C" __global__ void GetItemBuildIndices(const INDEX_TYPE* __restrict__ index,
                                               INDEX_TYPE* __restrict__ element_index,
                                               ERROR_TYPE* __restrict__ error,
                                               int32_t index_dim,
                                               int32_t indexCount,
                                               int32_t dim_size,
                                               tensor_view_5d_t index_tv,
                                               int32_t dim_offset,
                                               int32_t dim_info_offset)
{
    // instantiate the kernel
    getitembuildindices<INDEX_TYPE, ERROR_TYPE>(index,
                                                element_index,
                                                error,
                                                index_dim,
                                                indexCount,
                                                dim_size,
                                                index_tv,
                                                dim_offset,
                                                dim_info_offset);
}

extern "C" __global__ void GetitemBwd(const INPUT_TYPE* __restrict__ dy,
                                      INDEX_TYPE* __restrict__ element_index,
                                      OUTPUT_TYPE* __restrict__ dx,
                                      int32_t start_dim,
                                      int32_t indexCount,
                                      tensor_view_5d_t dy_tv,
                                      tensor_view_5d_t dx_tv,
                                      int32_t dim_info_offset,
                                      int32_t offset)
{
    // instantiate the kernel
    getitembwd<INPUT_TYPE, INDEX_TYPE, OUTPUT_TYPE>(
        dy, element_index, dx, start_dim, indexCount, dy_tv, dx_tv, dim_info_offset, offset);
}
