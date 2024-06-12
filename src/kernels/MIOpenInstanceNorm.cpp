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

#ifndef IN_OUT_TYPE
#define IN_OUT_TYPE float
#endif

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 256
#endif

template <typename TIO>
__device__ void instanceNormFwdTrain(const TIO* x,
                                     TIO* y,
                                     const TIO* scale,
                                     const TIO* bias,
                                     const TIO* running_mean_in,
                                     const TIO* running_var_in,
                                     TIO* running_mean_out,
                                     TIO* running_var_out,
                                     TIO* mean_var,
                                     float eps,
                                     float momentum,
                                     uint64_t outer_size,
                                     uint64_t inner_size,
                                     tensor_view_t<5> x_tv,
                                     tensor_view_t<5> y_tv,
                                     tensor_view_t<1> scale_tv,
                                     tensor_view_t<1> bias_tv,
                                     tensor_view_t<1> running_mean_in_tv,
                                     tensor_view_t<1> running_var_in_tv,
                                     tensor_view_t<1> running_mean_out_tv,
                                     tensor_view_t<1> running_var_out_tv,
                                     tensor_view_t<2> mean_var_tv)
{
    /*
     * Each group works on a single channel.
     * Example)
     * x dim = {N, C, L}
     * => gws = {C * LOCAL_SIZE}, lws = {LOCAL_SIZE}
     *    outer_size = N, target_size = C, inner_size = L
     */

    /*
     * Reduction to calculate mean and variance
     */
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;

    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE];

    FLOAT_ACCUM smean = static_cast<FLOAT_ACCUM>(0.0f);
    FLOAT_ACCUM svar  = static_cast<FLOAT_ACCUM>(0.0f);
    for(uint64_t i = 0; i < outer_size; ++i)
    {
        FLOAT_ACCUM pmean = static_cast<FLOAT_ACCUM>(0.0f);
        FLOAT_ACCUM pvar  = static_cast<FLOAT_ACCUM>(0.0f);

        for(uint64_t j = lid; j < inner_size; j += LOCAL_SIZE)
        {
            uint64_t xidx23 = j / x_tv.size[4], xidx4 = j % x_tv.size[4];
            uint64_t xidx2 = xidx23 / x_tv.size[3], xidx3 = xidx23 % x_tv.size[3];
            uint64_t xidx0 = i, xidx1 = gid;
            uint64_t xidx = x_tv.stride[4] * xidx4 + x_tv.stride[3] * xidx3 +
                            x_tv.stride[2] * xidx2 + x_tv.stride[1] * xidx1 +
                            x_tv.stride[0] * xidx0;
            FLOAT_ACCUM tmp = CVT_FLOAT2ACCUM(x[xidx]);
            pmean += tmp;
            pvar += tmp * tmp;
        }

        ltmp1[lid] = pmean;
        ltmp2[lid] = pvar;

        __syncthreads();
        for(uint32_t j = LOCAL_SIZE >> 1; j > 0; j >>= 1)
        {
            if(lid < j)
            {
                ltmp1[lid] += ltmp1[lid + j];
                ltmp2[lid] += ltmp2[lid + j];
            }
            __syncthreads();
        }

        pmean = ltmp1[0] / inner_size;
        pvar  = ltmp2[0] / inner_size - pmean * pmean;

        if(lid == 0)
        {
            if(mean_var)
            {
                mean_var[mean_var_tv.stride[1] * (gid * 2) + mean_var_tv.stride[0] * i] =
                    CVT_ACCUM2FLOAT(pmean);
                mean_var[mean_var_tv.stride[1] * (gid * 2 + 1) + mean_var_tv.stride[0] * i] =
                    CVT_ACCUM2FLOAT(rsqrt(pvar + eps));
            }
            smean += ltmp1[0];
            svar += ltmp2[0];
        }
        __syncthreads();

        FLOAT_ACCUM pscale = scale ? CVT_FLOAT2ACCUM(scale[scale_tv.stride[0] * gid])
                                   : static_cast<FLOAT_ACCUM>(1.0f);
        FLOAT_ACCUM pbias =
            bias ? CVT_FLOAT2ACCUM(bias[bias_tv.stride[0] * gid]) : static_cast<FLOAT_ACCUM>(0.0f);

        for(uint64_t j = lid; j < inner_size; j += LOCAL_SIZE)
        {
            uint64_t xidx23 = j / x_tv.size[4], xidx4 = j % x_tv.size[4];
            uint64_t xidx2 = xidx23 / x_tv.size[3], xidx3 = xidx23 % x_tv.size[3];
            uint64_t xidx0 = i, xidx1 = gid;
            uint64_t xidx = x_tv.stride[4] * xidx4 + x_tv.stride[3] * xidx3 +
                            x_tv.stride[2] * xidx2 + x_tv.stride[1] * xidx1 +
                            x_tv.stride[0] * xidx0;

            uint64_t yidx23 = j / y_tv.size[4], yidx4 = j % y_tv.size[4];
            uint64_t yidx2 = yidx23 / y_tv.size[3], yidx3 = yidx23 % y_tv.size[3];
            uint64_t yidx0 = i, yidx1 = gid;
            uint64_t yidx = y_tv.stride[4] * yidx4 + y_tv.stride[3] * yidx3 +
                            y_tv.stride[2] * yidx2 + y_tv.stride[1] * yidx1 +
                            y_tv.stride[0] * yidx0;
            y[yidx] = CVT_ACCUM2FLOAT(
                (CVT_FLOAT2ACCUM(x[xidx]) - pmean) * rsqrt(pvar + eps) * pscale + pbias);
        }
    }

    if(lid == 0)
    {
        if(running_mean_in)
        {
            smean = smean / (outer_size * inner_size);
            svar  = svar / (outer_size * inner_size) - smean * smean;
            running_mean_out[running_mean_out_tv.stride[0] * gid] = CVT_ACCUM2FLOAT(
                (1 - momentum) *
                    CVT_FLOAT2ACCUM(running_mean_in[running_mean_in_tv.stride[0] * gid]) +
                momentum * smean);
            running_var_out[running_var_out_tv.stride[0] * gid] = CVT_ACCUM2FLOAT(
                (1 - momentum) *
                    CVT_FLOAT2ACCUM(running_var_in[running_var_in_tv.stride[0] * gid]) +
                momentum * svar);
        }
    }
}

extern "C" __global__ void InstanceNormFwdTrain(const IN_OUT_TYPE* x,
                                                IN_OUT_TYPE* y,
                                                const IN_OUT_TYPE* scale,
                                                const IN_OUT_TYPE* bias,
                                                const IN_OUT_TYPE* running_mean_in,
                                                const IN_OUT_TYPE* running_var_in,
                                                IN_OUT_TYPE* running_mean_out,
                                                IN_OUT_TYPE* running_var_out,
                                                IN_OUT_TYPE* mean_var,
                                                float eps,
                                                float momentum,
                                                uint64_t outer_size,
                                                uint64_t inner_size,
                                                tensor_view_t<5> x_tv,
                                                tensor_view_t<5> y_tv,
                                                tensor_view_t<1> scale_tv,
                                                tensor_view_t<1> bias_tv,
                                                tensor_view_t<1> running_mean_in_tv,
                                                tensor_view_t<1> running_var_in_tv,
                                                tensor_view_t<1> running_mean_out_tv,
                                                tensor_view_t<1> running_var_out_tv,
                                                tensor_view_t<2> mean_var_tv)
{
    instanceNormFwdTrain<IN_OUT_TYPE>(x,
                                      y,
                                      scale,
                                      bias,
                                      running_mean_in,
                                      running_var_in,
                                      running_mean_out,
                                      running_var_out,
                                      mean_var,
                                      eps,
                                      momentum,
                                      outer_size,
                                      inner_size,
                                      x_tv,
                                      y_tv,
                                      scale_tv,
                                      bias_tv,
                                      running_mean_in_tv,
                                      running_var_in_tv,
                                      running_mean_out_tv,
                                      running_var_out_tv,
                                      mean_var_tv);
}