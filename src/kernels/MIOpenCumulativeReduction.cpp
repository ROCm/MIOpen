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

#include "float_types.h"
#include "tensor_view.hpp"
#include "MIOpenCumulativeReduction.hpp"

template <CumulativeReductionOp_t OP, uint64_t LOCAL_SIZE, typename T1, typename... T2>
__device__ inline void
CumulativeReductionScan(const int& lid, T1* __restrict__ a, T2* __restrict__... b)
{
    // reduction
    int stride = 1;
    while(stride <= LOCAL_SIZE)
    {
        int idx = (lid + 1) * stride * 2 - 1;
        if(idx < LOCAL_SIZE)
            reduce_func<OP, T1, T2...>{}.calculate(
                a[idx], a[idx - stride], b[idx]..., b[idx - stride]...);
        stride *= 2;
        __syncthreads();
    }

    // post scan
    stride = LOCAL_SIZE / 2;
    while(stride > 0)
    {
        int idx = (lid + 1) * stride * 2 - 1;
        if((idx + stride) < LOCAL_SIZE)
            reduce_func<OP, T1, T2...>{}.calculate(
                a[idx + stride], a[idx], b[idx + stride]..., b[idx]...);
        stride = stride / 2;
        __syncthreads();
    }
}

template <typename TI, typename TO, CumulativeReductionOp_t OP, int NDIMS, uint64_t LOCAL_SIZE>
__device__ void LocalCumulativeReduction(const TI* __restrict__ input,
                                         TO* __restrict__ local_output,
                                         int* __restrict__ local_indices,
                                         const unsigned int dim,
                                         const bool exclusive,
                                         const bool reverse,
                                         tensor_view_t<NDIMS> input_tv)
{
    static __shared__ FLOAT_ACCUM otmp[LOCAL_SIZE];
    int* itmp = nullptr;
    if(local_indices)
    {
        static __shared__ int _itmp[LOCAL_SIZE];
        itmp = _itmp;
    }

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t lid = threadIdx.y;

    // get input index
    size_t idx;
    tensor_layout_t<NDIMS> input_layout;
    {
        auto _gid = gid;
        for(int i = NDIMS - 1; i > dim; --i)
        {
            input_layout.layout[i] = _gid % input_tv.size[i];
            _gid /= input_tv.size[i];
        }

        auto local_inner_size    = (input_tv.size[dim] + LOCAL_SIZE - 1) / LOCAL_SIZE;
        idx                      = lid + _gid % local_inner_size * LOCAL_SIZE;
        input_layout.layout[dim] = (reverse ? input_tv.size[dim] - idx - 1 : idx);
        _gid /= local_inner_size;

        for(int i = dim - 1; i >= 0; --i)
        {
            input_layout.layout[i] = _gid % input_tv.size[i];
            _gid /= input_tv.size[i];
        }
    }

    if(idx < input_tv.size[dim])
    {
        otmp[lid] = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(input_layout)]);
        itmp[lid] = input_layout.layout[dim];
    }
    else
    {
        otmp[lid] = reduce_func<OP, FLOAT_ACCUM>{}.START_VAL;
        itmp[lid] = 0;
    }
    __syncthreads();

    for(size_t i = LOCAL_SIZE / 2; i > 0; i >>= 1)
    {
        if(lid < i)
            reduce_func<OP, FLOAT_ACCUM, int>{}.calculate(
                otmp[lid], otmp[lid + i], itmp[lid], itmp[lid + i]);
        __syncthreads();
    }

    if(lid == 0)
    {
        size_t inner_size = 1;
        for(int i = NDIMS - 1; i > dim; --i)
            inner_size *= input_tv.size[i];
        size_t reduce_size = (input_tv.size[dim] + LOCAL_SIZE - 1) / LOCAL_SIZE;
        auto _gid          = gid;
        auto local_idx     = 0;
        local_idx += _gid % inner_size * reduce_size;
        _gid /= inner_size;
        local_idx += _gid % reduce_size;
        _gid /= reduce_size;
        local_idx += _gid * reduce_size * inner_size;

        local_output[local_idx]  = CVT_ACCUM2FLOAT(otmp[0]);
        local_indices[local_idx] = itmp[0];
        // printf("LocalCumulativeReduction: %d %d %d %f %d\n",
        //        gid,
        //        idx,
        //        local_idx,
        //        local_output[local_idx],
        //        local_indices[local_idx]);
    }
}

extern "C" __global__ void LocalCumulativeReduction(const INPUT_TYPE* __restrict__ input,
                                                    OUTPUT_TYPE* __restrict__ local_output,
                                                    int* __restrict__ local_indices,
                                                    const unsigned int dim,
                                                    const bool exclusive,
                                                    const bool reverse,
                                                    tensor_view_t<VIEW_DIMS> input_tv)
{
    // instantiate the kernel
    LocalCumulativeReduction<INPUT_TYPE,
                             OUTPUT_TYPE,
                             (CumulativeReductionOp_t)OP_TYPE,
                             VIEW_DIMS,
                             REDUCE_SIZE>(
        input, local_output, local_indices, dim, exclusive, reverse, input_tv);
}

template <typename TI, typename TO, CumulativeReductionOp_t OP, uint64_t LOCAL_SIZE>
__device__ void CumulativeReductionNaiveForward(const TI* local_input,
                                                TO* local_output,
                                                int* __restrict__ local_indices,
                                                const size_t reduce_size,
                                                const size_t inner_size,
                                                const bool reverse)
{
    static __shared__ FLOAT_ACCUM otmp[LOCAL_SIZE];
    int* itmp = nullptr;
    if(local_indices)
    {
        static __shared__ int _itmp[LOCAL_SIZE];
        itmp = _itmp;
    }

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t lid = threadIdx.y;

    FLOAT_ACCUM tmp_val = reduce_func<OP, FLOAT_ACCUM>{}.START_VAL;
    int tmp_idx;

    for(int i = lid; i / LOCAL_SIZE <= (reduce_size - 1) / LOCAL_SIZE; i += LOCAL_SIZE)
    {
        // int idx = gid % inner_size + gid / inner_size * reduce_size * inner_size +
        //           (reverse ? reduce_size - i - 1 : i) * inner_size;
        int idx = gid * reduce_size + (reverse ? reduce_size - i - 1 : i);

        if(i < reduce_size)
        {
            otmp[lid] = CVT_FLOAT2ACCUM(local_input[idx]);
            if(local_indices)
                itmp[lid] = local_indices[idx];
            // printf("CumulativeReductionNaiveForward: %d %d %f %d\n",
            //        gid,
            //        idx,
            //        local_input[idx],
            //        local_indices[idx]);
        }
        else
        {
            otmp[lid] = reduce_func<OP, FLOAT_ACCUM>{}.START_VAL;
            if(local_indices)
                itmp[lid] = 0;
        }
        __syncthreads();

        if(local_indices)
            CumulativeReductionScan<OP, LOCAL_SIZE, FLOAT_ACCUM, int>(lid, otmp, itmp);
        else
            CumulativeReductionScan<OP, LOCAL_SIZE, FLOAT_ACCUM>(lid, otmp);

        if(i < reduce_size)
        {
            if(local_indices)
                reduce_func<OP, FLOAT_ACCUM, int>{}.calculate(
                    otmp[lid], tmp_val, itmp[lid], tmp_idx);
            else
                reduce_func<OP, FLOAT_ACCUM>{}.calculate(otmp[lid], tmp_val);
            if(local_output)
                local_output[idx] = CVT_ACCUM2FLOAT(otmp[lid]);
            if(local_indices)
                local_indices[idx] = itmp[lid];
            // printf("CumulativeReductionNaiveForward: %d %d %f %d\n",
            //        gid,
            //        idx,
            //        local_output[idx],
            //        local_indices[idx]);
        }
        if(local_indices)
            update<FLOAT_ACCUM, int>(tmp_val, otmp[LOCAL_SIZE - 1], tmp_idx, itmp[LOCAL_SIZE - 1]);
        else
            update<FLOAT_ACCUM>(tmp_val, otmp[LOCAL_SIZE - 1]);
        __syncthreads();
    }
}

extern "C" __global__ void CumulativeReductionNaiveForward(const INPUT_TYPE* local_input,
                                                           OUTPUT_TYPE* local_output,
                                                           int* __restrict__ local_indices,
                                                           const size_t reduce_size,
                                                           const size_t inner_size,
                                                           const bool reverse)
{
    // instantiate the kernel
    CumulativeReductionNaiveForward<INPUT_TYPE,
                                    OUTPUT_TYPE,
                                    (CumulativeReductionOp_t)OP_TYPE,
                                    REDUCE_SIZE>(
        local_input, local_output, local_indices, reduce_size, inner_size, reverse);
}

template <typename TI, typename TO, CumulativeReductionOp_t OP, int NDIMS, uint64_t LOCAL_SIZE>
__device__ void CumulativeReductionForward(const TI* __restrict__ input,
                                           TO* __restrict__ output,
                                           int* __restrict__ indices,
                                           const TO* __restrict__ local_output,
                                           const int* __restrict__ local_indices,
                                           const unsigned int dim,
                                           const bool exclusive,
                                           const bool reverse,
                                           tensor_view_t<NDIMS> input_tv,
                                           tensor_view_t<NDIMS> output_tv,
                                           tensor_view_t<NDIMS> indices_tv)
{
    static __shared__ FLOAT_ACCUM otmp[LOCAL_SIZE];
    int* itmp = nullptr;
    if(indices)
    {
        static __shared__ int _itmp[LOCAL_SIZE];
        itmp = _itmp;
    }

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t lid = threadIdx.y;

    size_t size = 1;
    for(int i = 0; i < NDIMS; ++i)
        size *= input_tv.size[i];
    size_t reduce_size = input_tv.size[dim];

    if(gid >= size / reduce_size)
        return;

    input_tv.size[dim] = 1;
    tensor_layout_t<NDIMS> tensor_layout(input_tv, gid);
    input_tv.size[dim] = reduce_size;
    int idx            = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < reduce_size)
    {
        tensor_layout.layout[dim] = (reverse ? reduce_size - idx - 1 : idx);
        otmp[lid] = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(tensor_layout)]);
        if(indices)
            itmp[lid] = tensor_layout.layout[dim];
    }
    else
    {
        otmp[lid] = reduce_func<OP, FLOAT_ACCUM>{}.START_VAL;
        if(indices)
            itmp[lid] = 0;
    }
    __syncthreads();

    if(indices)
        CumulativeReductionScan<OP, LOCAL_SIZE, FLOAT_ACCUM, int>(lid, otmp, itmp);
    else
        CumulativeReductionScan<OP, LOCAL_SIZE, FLOAT_ACCUM>(lid, otmp);

    if(idx < reduce_size)
    {
        FLOAT_ACCUM tmp_val = otmp[lid];
        int tmp_idx;
        if(indices)
            tmp_idx = itmp[lid];

        if(local_output)
        {
            if(blockIdx.y >= 1)
            {
                if(local_indices)
                {
                    int local_idx =
                        gid * ((reduce_size + LOCAL_SIZE - 1) / LOCAL_SIZE) + blockIdx.y - 1;
                    FLOAT_ACCUM olocal = CVT_FLOAT2ACCUM(local_output[local_idx]);
                    int ilocal         = local_indices[local_idx];
                    reduce_func<OP, FLOAT_ACCUM, int>{}.calculate(tmp_val, olocal, tmp_idx, ilocal);
                    // printf("%p %p %d %d %f %d %f %d\n",
                    //        local_output,
                    //        local_indices,
                    //        gid,
                    //        local_idx,
                    //        olocal,
                    //        ilocal,
                    //        tmp_val,
                    //        tmp_idx);
                }
                else
                {
                    reduce_func<OP, FLOAT_ACCUM>{}.calculate(
                        tmp_val,
                        CVT_FLOAT2ACCUM(
                            local_output[gid * ((reduce_size + LOCAL_SIZE - 1) / LOCAL_SIZE) +
                                         blockIdx.y - 1]));
                }
            }
        }

        if(indices)
            reduce_func<OP, FLOAT_ACCUM, int>{}.calculate(otmp[lid], tmp_val, itmp[lid], tmp_idx);
        else
            reduce_func<OP, FLOAT_ACCUM>{}.calculate(otmp[lid], tmp_val);

        if(output)
            output[output_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(tmp_val);
        if(indices)
            indices[indices_tv.get_tensor_view_idx(tensor_layout)] = tmp_idx;
    }
}

extern "C" __global__ void CumulativeReductionForward(const INPUT_TYPE* __restrict__ input,
                                                      OUTPUT_TYPE* __restrict__ output,
                                                      int* __restrict__ indices,
                                                      const OUTPUT_TYPE* __restrict__ local_output,
                                                      const int* __restrict__ local_indices,
                                                      const unsigned int dim,
                                                      const bool exclusive,
                                                      const bool reverse,
                                                      tensor_view_t<VIEW_DIMS> input_tv,
                                                      tensor_view_t<VIEW_DIMS> output_tv,
                                                      tensor_view_t<VIEW_DIMS> indices_tv)
{
    // instantiate the kernel
    CumulativeReductionForward<INPUT_TYPE,
                               OUTPUT_TYPE,
                               (CumulativeReductionOp_t)OP_TYPE,
                               VIEW_DIMS,
                               REDUCE_SIZE>(input,
                                            output,
                                            indices,
                                            local_output,
                                            local_indices,
                                            dim,
                                            exclusive,
                                            reverse,
                                            input_tv,
                                            output_tv,
                                            indices_tv);
}
