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
// #include <cmath>
#include <cstdio>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"

// __device__ inline FLOAT_ACCUM sign_(FLOAT_ACCUM val) { return (0 < val) - (val < 0); }
__device__ inline FLOAT_ACCUM sign_(FLOAT_ACCUM val)
{
    return static_cast<FLOAT_ACCUM>(0 < val) - static_cast<FLOAT_ACCUM>(val < 0);
}

__device__ inline FLOAT_ACCUM backward(const FLOAT_ACCUM diff,
                                       const FLOAT_ACCUM grad,
                                       const FLOAT_ACCUM dist,
                                       const FLOAT_ACCUM p)
{
    // printf("p: %f\n", p);

    if(p == 1.f)
    { // one
        return grad * sign_(diff);
    }
    else if(p < 2.f)
    { // lt_two

        return (dist == 0.0 || (diff == 0.0 && p < 1))
                   ? 0
                   : (sign_(diff) * pow(fabs(diff), p - 1) * grad / pow(dist, p - 1));
    }
    else if(p == 2.f)
    { // two
        return dist == 0.0 ? 0 : grad * diff / dist;
    }
    else if(isinf(p))
    { // inf
        // printf("[Kernel] Hit here infinity\n");
        // printf("[Kernel] diff: %f, dist: %f, fabs(diff) == dist: %f\n",
        //        diff,
        //        dist,
        //        static_cast<FLOAT_ACCUM>(fabs(diff) == dist));
        return grad * sign_(diff) * static_cast<FLOAT_ACCUM>(fabs(diff) == dist);
    }
    else
    { // p
        // printf("diff: %f, dist: %f, grad: %f, pow(fabs(diff), p - 2): %f, pow(dist, p - 1):
        // %f\n",
        //        diff,
        //        dist,
        //        grad,
        //        pow(fabs(diff), p - 2),
        //        pow(dist, p - 1));
        return dist == 0.0 ? 0 : diff * pow(fabs(diff), p - 2) * grad / pow(dist, p - 1);
    }
}

template <typename DTYPE>
__device__ void pdist_backward(const DTYPE* __restrict__ input,
                               const DTYPE* __restrict__ output,
                               const DTYPE* __restrict__ grad, // output_grad
                               DTYPE* __restrict__ input_grad,
                               double p,
                               double n2,
                               double n2_squared_minus_1,
                               tensor_view_t<2> input_tv,
                               tensor_view_t<1> output_tv,
                               tensor_view_t<1> grad_tv
                               //  tensor_view_t<3> input_grad_tv
)

{
    // NO = N(N-1)/2
    // gws = {NO * M}
    // input = {N, M}
    // output = {NO}
    // grad = {NO}
    // input_grad = {N - 1, N, M} = {NO * 2, M}
    // input_grad[0][j] = pair(A,x)

    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: pass those values as params to avoid redundant calculations
    long N  = input_tv.size[0];
    long NO = output_tv.size[0];
    long M  = input_tv.size[1];

    auto i_tl = tensor_layout_t<2>(input_tv, gid);

    auto k = i_tl.layout[0]; // output_pair_index
    auto m = i_tl.layout[1]; // column_index

    if(k >= NO)
        return;

    // printf("gid: %ld, k: %ld, m: %ld\n", gid, k, m);
    // pair(row_i, row_j) corresponding to output[k]
    // e.g. pair(point_A, point_B)
    long i = n2 - sqrt(n2_squared_minus_1 - 2 * k);
    long j = k - N * i + i * (i + 1) / 2 + i + 1;
    // printf("")
    // printf("gid: %ld, N: %ld, n2: %f, n2_squared_minus_1: %f, k: %ld, m: %ld, i: %ld, j: %ld\n",
    //        gid,
    //        N,
    //        n2,
    //        n2_squared_minus_1,
    //        k,
    //        m,
    //        i,
    //        j);

    // Pair of gradients corresponding to pair(i,j)
    // e.g. pair(input_grad(A,B), input_grad(B,A))
    // those 2 elements in pair have opposite signs
    // just similar to dist(A,B) vs. dist(B,A)
    long ib = j - i - 1;
    long jb = N - 2 - i;

    auto grad_tl   = tensor_layout_t<1>(grad_tv, k);
    auto output_tl = tensor_layout_t<1>(output_tv, k);

    // auto input_idx = input_tv.get_tensor_view_idx(i_tl);
    auto grad_idx   = grad_tv.get_tensor_view_idx(grad_tl);
    auto output_idx = output_tv.get_tensor_view_idx(output_tl);

    // printf("gid: %ld, grad_idx: %ld, output_idx: %ld\n", gid, grad_idx, output_idx);

    // DTYPE grad_k   = grad[grad_idx];
    // DTYPE output_k = output[output_idx];
    FLOAT_ACCUM grad_k   = CVT_FLOAT2ACCUM(grad[grad_idx]);
    FLOAT_ACCUM output_k = CVT_FLOAT2ACCUM(output[output_idx]);

    // printf("output_idx: %ld, grad_k: %f, output_k: %f\n", output_idx, grad_k, output_k);

    // i_tl =
    auto input_idx_0 = input_tv.get_tensor_view_idx({i, m});
    auto input_idx_1 = input_tv.get_tensor_view_idx({j, m});

    // printf("gid: %ld, i: %ld, j: %ld, m: %ld, input_idx_0: %ld, input_idx_1: %ld\n",
    //        gid,
    //        i,
    //        j,
    //        m,
    //        input_idx_0,
    //        input_idx_1);

    // printf("gid: %ld. input_idx_0: %ld, input_idx_1: %ld\n", gid, input_idx_0, input_idx_1);

    FLOAT_ACCUM diff = CVT_FLOAT2ACCUM(input[input_idx_0]) - CVT_FLOAT2ACCUM(input[input_idx_1]);

    // printf("gid: %ld. input_idx_0: %ld, input_idx_1: %ld, input[input_idx_0]: %f,
    // input[input_idx_1]: %f\n", gid, input_idx_0, input_idx_1,
    // CVT_FLOAT2ACCUM(input[input_idx_0]), CVT_FLOAT2ACCUM(input[input_idx_1]));

    // FLOAT_ACCUM res = backward(diff, grad_k, output_k, p_);

    FLOAT_ACCUM p_ = static_cast<FLOAT_ACCUM>(p);
    // DTYPE res      = CVT_ACCUM2FLOAT(backward(diff, grad_k, output_k, p_));

    // input_grad[ib * N * M + i * M + m] = res;  // dist(i,j)
    // input_grad[jb * N * M + j * M + m] = -res; // dist(j,i)

    // auto input_grad_idx_0 = input_grad_tv.get_tensor_view_idx({ib, i, m});
    // auto input_grad_idx_1 = input_grad_tv.get_tensor_view_idx({jb, j, m});

    // input_grad[input_grad_idx_0] = res;
    // input_grad[input_grad_idx_1] = -res;
    // printf("gid: %ld, k: %ld, input_idx_0: %ld, input_idx_1: %ld, input_grad_idx_0: %ld,
    // input_grad_idx_1: %ld, res: %f\n", gid, k, input_idx_0, input_idx_1, ib * N * M + i * M + m,
    // jb * N * M + j * M + m, res);

    // printf("[Kernel] ib: %d, i: %d, jb: %d, j: %d, m: %d, res: %f\n", ib, i, jb, j, m, res);

    // printf("gid: %ld, k: %ld, input_grad_idx_0: %ld, input_grad_idx_1, res: %f\n",
    //        gid,
    //        k,
    //        ib * N * M + i * M + m,
    //        jb * N * M + j * M + m,
    //        res);
    // input_grad[ib * N * M + i * M + m] =
    // CVT_ACCUM2FLOAT(res); input_grad[jb * N * M + j * M + m] = -CVT_ACCUM2FLOAT(res);

    FLOAT_ACCUM res = backward(diff, grad_k, output_k, p_);

    // if(isinf(res)) {
    //     printf("res is inf\n");
    //     printf("res: %f\n", res);
    // }

    // auto tmp = CVT_ACCUM2FLOAT(res);
    // auto tmp_check = CVT_FLOAT2ACCUM(tmp);

    // if (isinf(tmp_check)) {
    //     printf("converted res is inf\n");
    //     printf("diff: %f, grad_k: %f, output_k: %f, original_res: %f\n", diff, grad_k, output_k,
    //     res);
    //     // printf("converted res: %f\n", tmp_check);
    // }

    input_grad[ib * N * M + i * M + m] = CVT_ACCUM2FLOAT(res);  // dist(i,j)
    input_grad[jb * N * M + j * M + m] = CVT_ACCUM2FLOAT(-res); // dist(j,i)
}

extern "C" __global__ void PdistBackward(const INPUT_TYPE* __restrict__ input,
                                         const INPUT_TYPE* __restrict__ output,
                                         const INPUT_TYPE* __restrict__ grad, // output_grad
                                         INPUT_TYPE* __restrict__ input_grad,
                                         double p,
                                         double n2,
                                         double n2_squared_minus_1,
                                         tensor_view_t<2> input_tv,
                                         tensor_view_t<1> output_tv,
                                         tensor_view_t<1> grad_tv
                                         //  tensor_view_t<3> input_grad_tv
)
{
    // printf("p_: %f\n", p_);
    pdist_backward<INPUT_TYPE>(
        input, output, grad, input_grad, p, n2, n2_squared_minus_1, input_tv, output_tv, grad_tv
        //    input_grad_tv
    );
}
