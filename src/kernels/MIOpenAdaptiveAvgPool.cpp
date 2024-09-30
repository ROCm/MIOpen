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
#include <cstddef>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE float
#endif

template <typename TI, typename TO>
__device__ void avgPoolForward1d(const TI* __restrict__ input,
                                 TO* __restrict__ output,
                                 size_t N,
                                 size_t C,
                                 size_t H,
                                 size_t OH,
                                 tensor_view_t<3> input_tv,
                                 tensor_view_t<3> output_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t nc = gid / OH, oh = gid % OH;
    size_t n = nc / C, c = nc % C;
    if(n >= N)
        return;

    int32_t h  = (int32_t)floor((float)(oh * H) / OH);
    int32_t kh = (int32_t)ceil((float)((oh + 1) * H) / OH) - h;

    DTYPE_ACCURATE sum = 0;
    for(int ih = h; ih < (h + kh); ++ih)
    {
        sum += GET_3D_VAL_AT(input, n, c, ih);
    }

    SET_3D_VAL_AT(output, n, c, oh, sum / kh);
}
extern "C" __global__ void AvgPoolForward1d(const INPUT_TYPE* __restrict__ input,
                                            OUTPUT_TYPE* __restrict__ output,
                                            size_t N,
                                            size_t C,
                                            size_t H,
                                            size_t OH,
                                            tensor_view_t<3> input_tv,
                                            tensor_view_t<3> output_tv)
{
    avgPoolForward1d<INPUT_TYPE, OUTPUT_TYPE>(input, output, N, C, H, OH, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolBackward1d(const TI* __restrict__ output_grad,
                                  TO* __restrict__ input_grad,
                                  size_t N,
                                  size_t C,
                                  size_t H,
                                  size_t OH,
                                  tensor_view_t<3> output_grad_tv,
                                  tensor_view_t<3> input_grad_tv)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t nc = gid / H, h = gid % H;
    size_t n = nc / C, c = nc % C;
    if(n >= N)
        return;

    int32_t oh  = (int32_t)floor((float)(h * OH) / H);
    int32_t koh = (int32_t)ceil((float)((h + 1) * OH) / H) - oh;

    DTYPE_ACCURATE grad = 0;
    for(int ih = oh; ih < (oh + koh); ++ih)
    {
        int32_t kh =
            (int32_t)ceil((float)((ih + 1) * H) / OH) - (int32_t)floor((float)(ih * H) / OH);
        grad += GET_3D_VAL_AT(output_grad, n, c, ih) / kh;
    }

    SET_3D_VAL_AT(input_grad, n, c, h, grad);
}
extern "C" __global__ void AvgPoolBackward1d(const INPUT_TYPE* __restrict__ output_grad,
                                             OUTPUT_TYPE* __restrict__ input_grad,
                                             size_t N,
                                             size_t C,
                                             size_t H,
                                             size_t OH,
                                             tensor_view_t<3> output_grad_tv,
                                             tensor_view_t<3> input_grad_tv)
{
    avgPoolBackward1d<INPUT_TYPE, OUTPUT_TYPE>(
        output_grad, input_grad, N, C, H, OH, output_grad_tv, input_grad_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolForward2d(const TI* __restrict__ input,
                                 TO* __restrict__ output,
                                 size_t N,
                                 size_t C,
                                 size_t H,
                                 size_t W,
                                 size_t OH,
                                 size_t OW,
                                 tensor_view_t<4> input_tv,
                                 tensor_view_t<4> output_tv)
{
    int32_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t ncoh = gid / OW, ow = gid % OW;
    int32_t nc = ncoh / OH, oh = ncoh % OH;
    int32_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    size_t h  = (size_t)floor((float)(oh * H) / OH);
    size_t kh = (size_t)ceil((float)((oh + 1) * H) / OH) - h;

    size_t w  = (size_t)floor((float)(ow * W) / OW);
    size_t kw = (size_t)ceil((float)((ow + 1) * W) / OW) - w;

    FSTYPE divider = (FSTYPE)(kh * kw);
    FSTYPE sum     = 0;
    for(size_t ih = h; ih < (h + kh); ++ih)
    {
        for(size_t iw = w; iw < (w + kw); ++iw)
        {
            sum += GET_4D_VAL_AT(input, n, c, ih, iw);
        }
    }

    SET_4D_VAL_AT(output, n, c, oh, ow, sum / divider);

    output[output_tv.get_tensor_view_idx(tensor_layout_t<4>(n, c, oh, ow))] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void AvgPoolForward2d(const INPUT_TYPE* __restrict__ input,
                                            OUTPUT_TYPE* __restrict__ output,
                                            size_t N,
                                            size_t C,
                                            size_t H,
                                            size_t W,
                                            size_t OH,
                                            size_t OW,
                                            tensor_view_t<4> input_tv,
                                            tensor_view_t<4> output_tv)
{
    avgPoolForward2d<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, N, C, H, W, OH, OW, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolBackward2d(const TI* __restrict__ output_grad,
                                  TO* __restrict__ input_grad,
                                  size_t N,
                                  size_t C,
                                  size_t H,
                                  size_t W,
                                  size_t OH,
                                  size_t OW,
                                  tensor_view_t<4> output_grad_tv,
                                  tensor_view_t<4> input_grad_tv)
{
    int32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t nch = gid / W, w = gid % W;
    int32_t nc = nch / H, h = nch % H;
    int32_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    size_t oh  = (size_t)floor((float)(h * OH) / H);
    size_t koh = (size_t)ceil((float)((h + 1) * OH) / H) - oh;

    size_t ow  = (size_t)floor((float)(w * OW) / W);
    size_t kow = (size_t)ceil((float)((w + 1) * OW) / W) - ow;

    FLOAT_ACCUM grad = 0;
    for(size_t ih = oh; ih < (oh + koh); ++ih)
    {
        size_t kh = (size_t)ceil((float)((ih + 1) * H) / OH) - (size_t)floor((float)(ih * H) / OH);
        for(size_t iw = ow; iw < (ow + kow); ++iw)
        {
            size_t kw =
                (size_t)ceil((float)((iw + 1) * W) / OW) - (size_t)floor((float)(iw * W) / OW);
            grad += (FSTYPE)(GET_4D_VAL_AT(output_grad, n, c, ih, iw)) / (kh * kw);
        }
    }

    SET_4D_VAL_AT(input_grad, n, c, h, w, grad);

    input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout_t<4>(n, c, h, w))] =
        CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AvgPoolBackward2d(const INPUT_TYPE* __restrict__ output_grad,
                                             OUTPUT_TYPE* __restrict__ input_grad,
                                             size_t N,
                                             size_t C,
                                             size_t H,
                                             size_t W,
                                             size_t OH,
                                             size_t OW,
                                             tensor_view_t<4> output_grad_tv,
                                             tensor_view_t<4> input_grad_tv)
{
    avgPoolBackward2d<INPUT_TYPE, OUTPUT_TYPE>(
        output_grad, input_grad, N, C, H, W, OH, OW, output_grad_tv, input_grad_tv);
}

// __kernel void AdaptiveAvgpool2dBackward1x1OutputNHWC(const __global DTYPE_PTR output_grad,
//                                                      __global DTYPE_PTR input_grad,
//                                                      const int32_t N,
//                                                      const int32_t C,
//                                                      const int32_t HW,
//                                                      const int32_t output_grad_off,
//                                                      const int32_t input_grad_off)
// {
// /* VSIZE 2 and 16 is fastest but don't know why */
// #define VSIZE 2
//     size_t gid = get_global_id(0) * VSIZE;
//     size_t c   = gid % C;
//     size_t n   = gid / C;
//     if(n >= N)
//         return;

//     __global DTYPE_VEC_PTR(VSIZE) output_grad_vec =
//         (__global DTYPE_VEC_PTR(VSIZE))(output_grad + n * C + c + output_grad_off);

//     DTYPE_VEC(VSIZE) output_grad_v = GET(output_grad_vec, 0) / HW;

//     __global DTYPE_VEC_PTR(VSIZE) input_grad_vec =
//         (__global DTYPE_VEC_PTR(VSIZE))(input_grad + n * C * HW + c + input_grad_off);

//     for(size_t i = 0; i < HW; ++i)
//     {
//         SET(input_grad_vec, i * C / VSIZE, output_grad_v);
//     }
// #undef VSIZE
// }

template <typename TI, typename TO>
__device__ void avgPoolForward3d(const TI* __restrict__ input,
                                 TO* __restrict__ output,
                                 size_t N,
                                 size_t C,
                                 size_t D,
                                 size_t H,
                                 size_t W,
                                 size_t OD,
                                 size_t OH,
                                 size_t OW,
                                 tensor_view_t<5> input_tv,
                                 tensor_view_t<5> output_tv)
{
    int32_t gid    = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t ncodoh = gid / OW, ow = gid % OW;
    int32_t ncod = ncodoh / OH, oh = ncodoh % OH;
    int32_t nc = ncod / OD, od = ncod % OD;
    int32_t n = nc / C, c = nc % C;

    if(n >= N)
        return;
    int32_t d  = (int32_t)floor((float)(od * D) / OD);
    int32_t kd = (int32_t)ceil((float)((od + 1) * D) / OD) - d;

    int32_t h  = (int32_t)floor((float)(oh * H) / OH);
    int32_t kh = (int32_t)ceil((float)((oh + 1) * H) / OH) - h;

    int32_t w  = (int32_t)floor((float)(ow * W) / OW);
    int32_t kw = (int32_t)ceil((float)((ow + 1) * W) / OW) - w;

    DTYPE_ACCURATE sum = 0;
    for(int32_t id = d; id < (d + kd); ++id)
    {
        for(int32_t ih = h; ih < (h + kh); ++ih)
        {
            for(int32_t iw = w; iw < (w + kw); ++iw)
            {
                sum += GET_5D_VAL_AT(input, n, c, id, ih, iw);
            }
        }
    }

    output[output_tv.get_tensor_view_idx(tensor_layout_t<5>(n, c, od, oh, ow))] =
        CVT_ACCUM2FLOAT(val);
    SET_5D_VAL_AT(output, n, c, od, oh, ow, sum / (kd * kh * kw));
}

extern "C" __global__ void AvgPoolForward3d(const INPUT_TYPE* __restrict__ input,
                                            OUTPUT_TYPE* __restrict__ output,
                                            size_t N,
                                            size_t C,
                                            size_t D,
                                            size_t H,
                                            size_t W,
                                            size_t OD,
                                            size_t OH,
                                            size_t OW,
                                            tensor_view_t<5> input_tv,
                                            tensor_view_t<5> output_tv)
{
    avgPoolForward3d<INPUT_TYPE, OUTPUT_TYPE>(
        input, output, N, C, D, H, W, OD, OH, OW, input_tv, output_tv);
}

template <typename TI, typename TO>
__device__ void avgPoolBackward3d(const TI* __restrict__ output_grad,
                                  TO* __restrict__ input_grad,
                                  size_t N,
                                  size_t C,
                                  size_t D,
                                  size_t H,
                                  size_t W,
                                  size_t OD,
                                  size_t OH,
                                  size_t OW,
                                  tensor_view_t<5> output_grad_tv,
                                  tensor_view_t<5> input_grad_tv)
{
    int32_t gid  = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t ncdh = gid / W, w = gid % W;
    int32_t ncd = ncdh / H, h = ncdh % H;
    int32_t nc = ncd / D, d = ncd % D;
    int32_t n = nc / C, c = nc % C;

    if(n >= N)
        return;

    int32_t od  = (int32_t)floor((float)(d * OD) / D);
    int32_t kod = (int32_t)ceil((float)((d + 1) * OD) / D) - od;

    int32_t oh  = (int32_t)floor((float)(h * OH) / H);
    int32_t koh = (int32_t)ceil((float)((h + 1) * OH) / H) - oh;

    int32_t ow  = (int32_t)floor((float)(w * OW) / W);
    int32_t kow = (int32_t)ceil((float)((w + 1) * OW) / W) - ow;

    DTYPE_ACCURATE grad = 0;
    for(int32_t id = od; id < (od + kod); ++id)
    {
        int32_t kd =
            (int32_t)ceil((float)((id + 1) * D) / OD) - (int32_t)floor((float)(id * D) / OD);
        for(int32_t ih = oh; ih < (oh + koh); ++ih)
        {
            int32_t kh =
                (int32_t)ceil((float)((ih + 1) * H) / OH) - (int32_t)floor((float)(ih * H) / OH);
            for(int32_t iw = ow; iw < (ow + kow); ++iw)
            {
                int32_t kw = (int32_t)ceil((float)((iw + 1) * W) / OW) -
                             (int32_t)floor((float)(iw * W) / OW);
                grad += GET_5D_VAL_AT(output_grad, n, c, id, ih, iw) / (kd * kh * kw);
            }
        }
    }

    input_grad[input_grad_tv.get_tensor_view_idx(tensor_layout_t<5>(n, c, d, h, w))] =
        CVT_ACCUM2FLOAT(grad);
}

extern "C" __global__ void AvgPoolBackward3d(const INPUT_TYPE* __restrict__ output_grad,
                                             OUTPUT_TYPE* __restrict__ input_grad,
                                             size_t N,
                                             size_t C,
                                             size_t D,
                                             size_t H,
                                             size_t W,
                                             size_t OD,
                                             size_t OH,
                                             size_t OW,
                                             tensor_view_t<5> output_grad_tv,
                                             tensor_view_t<5> input_grad_tv)
{
    avgPoolBackward3d<INPUT_TYPE, OUTPUT_TYPE>(
        output_grad, input_grad, N, C, D, H, W, OD, OH, OW, output_grad_tv, input_grad_tv);
}
