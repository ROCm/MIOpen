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

template <typename TI, typename TO, CumulativeReductionOp_t op, uint32_t NDIMS, uint32_t LOCAL_SIZE>
__device__ void CumulativeReductionNaiveFwdNd(const TI* __restrict__ input,
                                              TO* __restrict__ output,
                                              uint64_t* __restrict__ indices,
                                              const uint64_t dim,
                                              const bool exclusive,
                                              const bool reverse,
                                              tensor_view_t<NDIMS> input_tv,
                                              tensor_view_t<NDIMS> output_tv,
                                              tensor_view_t<NDIMS> indices_tv)
{
    static __shared__ FLOAT_ACCUM otmp[LOCAL_SIZE];
    uint64_t* itmp = nullptr;
    if(indices)
        static __shared__ uint64_t _itmp[LOCAL_SIZE];

    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t lid = threadIdx.y;

    size_t size = 1;
    for(int i = 0; i < NDIMS; ++i)
        size *= input_tv.size[i];
    size_t nelem = input_tv.size[dim];

    if(gid >= size / nelem)
        return;

    auto op_worker = reduce_func<op, FLOAT_ACCUM>{};

    tensor_layout_t<NDIMS> layout(input_tv, gid * nelem);
    FLOAT_ACCUM tmp_val = op_worker.START_VAL;
    uint64_t tmp_idx    = 0;

    for(int i = lid; i < nelem + LOCAL_SIZE - 1; i += LOCAL_SIZE)
    {
        layout.layout[dim] = (reverse ? input_tv.size[dim] - i - 1 : i);

        if(i < nelem)
        {
            if(exclusive)
            {
                if(layout.layout[dim] == 0)
                {
                    if(indices)
                        itmp[lid] = 0;
                    otmp[lid] = op_worker.START_VAL;
                }
                else
                {
                    layout.layout[dim] -= 1;
                    if(indices)
                        itmp[lid] = layout.layout[dim];
                    otmp[lid] = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(layout)]);
                    layout.layout[dim] += 1;
                }
            }
            else
            {
                if(indices)
                    itmp[lid] = layout.layout[dim];
                otmp[lid] = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(layout)]);
            }
        }
        else
        {
            if(indices)
                itmp[lid] = 0;
            otmp[lid] = op_worker.START_VAL;
        }
        __syncthreads();

        // reduction
        int stride = 1;
        while(stride <= LOCAL_SIZE)
        {
            int idx = (lid + 1) * stride * 2 - 1;
            if(idx < LOCAL_SIZE)
            {
                if(indices)
                    op_worker.calculate(
                        otmp[idx], otmp[idx - stride], itmp[idx], itmp[idx - stride]);
                else
                    op_worker.calculate(otmp[idx], otmp[idx - stride]);
            }
            stride *= 2;
            __syncthreads();
        }

        // post scan
        stride = LOCAL_SIZE / 2;
        while(stride > 0)
        {
            int idx = (lid + 1) * stride * 2 - 1;
            if((idx + stride) < LOCAL_SIZE)
            {
                if(indices)
                    op_worker.calculate(
                        otmp[idx + stride], otmp[idx], itmp[idx + stride], itmp[idx]);
                else
                    op_worker.calculate(otmp[idx + stride], otmp[idx]);
            }
            stride = stride / 2;
            __syncthreads();
        }

        if(i < nelem)
        {
            if(op_worker.isbetter(tmp_val, otmp[lid]))
            {
                if(output)
                    output[output_tv.get_tensor_view_idx(layout)] = CVT_ACCUM2FLOAT(tmp_val);
                if(indices)
                    indices[indices_tv.get_tensor_view_idx(layout)] = CVT_ACCUM2FLOAT(tmp_idx);
            }
            else
            {
                if(output)
                    output[output_tv.get_tensor_view_idx(layout)] = CVT_ACCUM2FLOAT(otmp[lid]);
                if(indices)
                    indices[indices_tv.get_tensor_view_idx(layout)] = CVT_ACCUM2FLOAT(itmp[lid]);
            }
        }
        if(indices)
            op_worker.calculate(tmp_val, otmp[LOCAL_SIZE - 1], tmp_idx, itmp[LOCAL_SIZE - 1]);
        else
            op_worker.calculate(tmp_val, otmp[LOCAL_SIZE - 1]);
        __syncthreads();
    }
}

extern "C" __global__ void CumulativeReductionNaiveFwdNd(const INPUT_TYPE* __restrict__ input,
                                                         OUTPUT_TYPE* __restrict__ output,
                                                         uint64_t* __restrict__ indices,
                                                         const uint64_t dim,
                                                         const bool exclusive,
                                                         const bool reverse,
                                                         tensor_view_t<VIEW_DIMS> input_tv,
                                                         tensor_view_t<VIEW_DIMS> output_tv,
                                                         tensor_view_t<VIEW_DIMS> indices_tv)
{
    // instantiate the kernel
    CumulativeReductionNaiveFwdNd<INPUT_TYPE, OUTPUT_TYPE, OP_TYPE, VIEW_DIMS, REDUCE_SIZE>(
        input, output, indices, dim, exclusive, reverse, input_tv, output_tv, indices_tv);
}
