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
#include "tensor_view.hpp"

/*
// Re-Normalize if norm > max_norm
extern "C" __global__ void EmbeddingReNorm(const long* input,
                                           FLOAT* weight,
                                           tensor_view_t<2> weight_tv,
                                           bool* index_error,
                                           FLOAT max_norm,
                                           FLOAT norm_type,
                                           long num_embeddings,
                                           int embedding_dim)
{
    int group_id = hipBlockIdx_x;
    int lid      = hipThreadIdx_x;

    __shared__ FLOAT ltmp[LOCAL_SIZE];

    long embedding_idx = input[group_id];

    if(embedding_idx < 0 || embedding_idx >= num_embeddings)
    {
        if(index_error)
            index_error[0] = true;
        return;
    }

    FLOAT norm = 0;
    tensor_layout_t<2> weight_idx{embedding_idx, 0};
    for(int i = lid; i < embedding_dim; i += LOCAL_SIZE)
    {
        weight_idx.layout[1] = i;
        auto val             = weight[weight_tv.get_tensor_view_idx(weight_idx)];
        norm += fabs(pow(val, norm_type));
    }

    ltmp[lid] = norm;
    __syncthreads();
    for(int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }

    if(lid == 0)
    {
        ltmp[0] = pow(ltmp[0], 1 / norm_type);
    }
    __syncthreads();

    if(ltmp[0] > max_norm)
    {
        FLOAT factor = max_norm / (ltmp[0] + 1e-7f);
        for(int i = lid; i < embedding_dim; i += LOCAL_SIZE)
        {
            tensor_layout_t<2> weight_idx{embedding_idx, i};
            weight[weight_tv.get_tensor_view_idx(weight_idx)] *= factor;
        }
    }
}
*/
Extern "C" __global__ void EmbeddingForward(long* input,
                                            FLOAT* weight,
                                            FLOAT* output,
                                            tensor_view_t<4> input_tv,
                                            tensor_view_t<2> weight_tv,
                                            tensor_view_t<5> output_tv,
                                            bool* index_error,
                                            long num_embeddings)
{
    size_t gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    tensor_layout_t<5> idx(output_tv, gid);

    if(n0 >= output_tv.size[0])
        return;

    size_t input_idx   = input_tv.get_tensor_view_idx(idx);
    long embedding_idx = input[input_idx];
    size_t output_idx  = output_tv.get_tensor_view_idx(idx);

    if(embedding_idx >= 0 && embedding_idx < num_embeddings)
    {
        tensor_layout_t<2> weight_idx{embedding_idx, idx.layout[4]};
        output[output_idx] = weight[weight_tv.get_tensor_view_idx(weight_idx)];
    }
    else
    {
        if(index_error)
            index_error[0] = true;
    }
}
/*
extern "C" __global__ void EmbeddingBackward(long* input,
                                             FLOAT* output_grad,
                                             FLOAT* weight_grad,
                                             int* indices_freq,
                                             bool* index_error,
                                             int embedding_dim,
                                             int input_size,
                                             tensor_view_t<4> input_tv,
                                             tensor_view_t<4> output_grad_tv,
                                             tensor_view_t<2> weight_grad_tv,
                                             long num_embeddings,
                                             long padding_idx)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= embedding_dim)
        return;

    for(int i = 0; i < input_size; ++i)
    {
        tensor_layout_t<4> input_layout{input_tv, i};
        auto input_idx     = input_tv.get_tensor_view_idx(input_layout);
        long embedding_idx = input[input_idx];

        if(embedding_idx == padding_idx)
            continue;

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            FLOAT scale = indices_freq
                              ? (static_cast<FLOAT>(1.0) / static_cast<FLOAT>(indices_freq[i]))
                              : 1.0f;

            tensor_layout_t<2> weight_layout{embedding_idx, gid};
            tensor_layout_t<2> output_layout{output_grad_tv, embedding_dim * i + gid};
            auto weight_idx = weight_grad_tv.get_tensor_view_idx(weight_layout);
            auto output_idx = output_grad_tv.get_tensor_view_idx(output_layout);
            weight_grad[weight_idx] += output_grad[output_idx] * scale;
        }
        else
        {
            if(index_error)
                index_error[0] = true;
        }
    }
}
*/
extern "C" __global__ void EmbeddingBackwardContiguous(long* input,
                                                       FLOAT* output_grad,
                                                       FLOAT* weight_grad,
                                                       int* indices_freq,
                                                       bool* index_error,
                                                       int embedding_dim,
                                                       int input_size,
                                                       long num_embeddings,
                                                       long padding_idx)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= embedding_dim)
        return;

    for(int i = 0; i < input_size; ++i)
    {
        long embedding_idx = input[i];

        if(embedding_idx == padding_idx)
            continue;

        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            FLOAT scale          = indices_freq
                                       ? (static_cast<FLOAT>(1.0) / static_cast<FLOAT>(indices_freq[i]))
                                       : 1.0;
            long weight_grad_idx = embedding_idx * embedding_dim + gid;
            long output_grad_idx = embedding_dim * i + gid;

            weight_grad[weight_grad_idx] += output_grad[output_grad_idx] * scale;
        }
        else
        {
            if(index_error)
                index_error[0] = true;
        }
    }
}
/*
extern "C" __global__ void EmbeddingBackwardAtomic(long* input,
                                                   FLOAT* output_grad,
                                                   FLOAT* weight_grad,
                                                   int* indices_freq,
                                                   bool* index_error,
                                                   int embedding_dim,
                                                   int input_size,
                                                   tensor_view_t<4> input_tv,
                                                   tensor_view_t<4> output_grad_tv,
                                                   tensor_view_t<2> weight_grad_tv,
                                                   long num_embeddings,
                                                   long padding_idx)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid / embedding_dim, j = gid % embedding_dim;
    if(i >= input_size)
        return;

    size_t n3 = i % input_tv.size[3], n012 = i / input_tv.size[3];
    size_t n2 = n012 % input_tv.size[2], n01 = n012 / input_tv.size[2];
    size_t n1 = n01 % input_tv.size[1], n0 = n01 / input_tv.size[1];

    size_t input_idx   = TV4D_IDX(input_tv, n0, n1, n2, n3);
    long embedding_idx = input[input_idx];

    if(embedding_idx == padding_idx)
        return;

    if(embedding_idx >= 0 && embedding_idx < num_embeddings)
    {
        FLOAT scale =
            indices_freq ? (static_cast<FLOAT>(1.0f) / static_cast<FLOAT>(indices_freq[i])) : 1.0f;

        atomic_add_g(&weight_grad[TV2D_IDX(weight_grad_tv, embedding_idx, j)],
                     GET_4D_VAL(output_grad, embedding_dim * i + j) * scale);
    }
    else
    {
        if(index_error)
            index_error[0] = true;
    }
}
*/
extern "C" __global__ void EmbeddingBackwardContiguousAtomic(long* input,
                                                             FLOAT* output_grad,
                                                             FLOAT* weight_grad,
                                                             int* indices_freq,
                                                             bool* index_error,
                                                             int embedding_dim,
                                                             int input_size,
                                                             long num_embeddings,
                                                             long padding_idx)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid / embedding_dim, j = gid % embedding_dim;
    if(i >= input_size)
        return;

    long embedding_idx = input[i];

    if(embedding_idx == padding_idx)
        return;

    if(embedding_idx >= 0 && embedding_idx < num_embeddings)
    {
        FLOAT scale =
            indices_freq ? (static_cast<FLOAT>(1.0f) / static_cast<FLOAT>(indices_freq[i])) : 1.0f;

        long weight_grad_idx = embedding_idx * embedding_dim + j;
        long output_grad_idx = embedding_dim * i + j;
        atomic_add_g(&weight_grad[weight_grad_idx], output_grad[output_grad_idx] * scale);
    }
    else
    {
        if(index_error)
            index_error[0] = true;
    }
}
/*
extern "C" __global__ void
EmbeddingBackwardSmallNumEmbeddingsTraverse(long* input,
                                            FLOAT* output_grad,
                                            FLOAT* weight_grad,
                                            int* indices_freq,
                                            bool* index_error,
                                            int embedding_dim,
                                            int input_size,
                                            tensor_view_t<4> input_tv,
                                            tensor_view_t<4> output_grad_tv,
                                            tensor_view_t<2> weight_grad_tv,
                                            long num_embeddings,
                                            long padding_idx)
{
    int gid                   = blockIdx.x * blockDim.x + threadIdx.x;
    int embedding_size        = num_embeddings * embedding_dim;
    int i                     = gid / embedding_size;
    int inner_embedding_space = gid % embedding_size;
    int target_embedding_idx  = inner_embedding_space / embedding_dim;
    int j                     = inner_embedding_space % embedding_dim;
    if(i >= input_size)
        return;

    FLOAT weight_grad_sum = 0;

    for(; i < input_size; i += ALPHA)
    {
        tensor_layout_t<4> input_layout{input_tv, i};
        size_t input_idx   = input_tv.get_tensor_view_idx(input_layout);
        long embedding_idx = input[input_idx];

        if(embedding_idx == padding_idx)
            continue;
        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            if(embedding_idx == target_embedding_idx)
            {
                FLOAT scale = indices_freq
                                  ? (static_cast<FLOAT>(1.0f) / static_cast<FLOAT>(indices_freq[i]))
                                  : 1.0f;
                weight_grad_sum += GET_4D_VAL(output_grad, embedding_dim * i + j) * scale;
            }
        }
        else
        {
            if(index_error)
                index_error[0] = true;
        }
    }

    tensor_layout_t<2> weight_grad_layout;
    weight_grad_layout.layout[0] = target_embedding_idx;
    weight_grad_layout.layout[1] = j;
    size_t weight_grad_idx       = weight_tv.get_tensor_view_idx(weight_grad_layout);
    atomic_add_g(&weight_grad[weight_grad_idx], weight_grad_sum);
}
*/
extern "C" __global__ void
EmbeddingBackwardSmallNumEmbeddingsTraverseContiguous(long* input,
                                                      FLOAT* output_grad,
                                                      FLOAT* weight_grad,
                                                      int* indices_freq,
                                                      bool* index_error,
                                                      int embedding_dim,
                                                      int input_size,
                                                      long num_embeddings,
                                                      long padding_idx)
{
    int gid                   = blockIdx.x * blockDim.x + threadIdx.x;
    int embedding_size        = num_embeddings * embedding_dim;
    int i                     = gid / embedding_size;
    int inner_embedding_space = gid % embedding_size;
    int target_embedding_idx  = inner_embedding_space / embedding_dim;
    int j                     = inner_embedding_space % embedding_dim;
    if(i >= input_size)
        return;

    FLOAT weight_grad_sum = 0;

    for(; i < input_size; i += ALPHA)
    {
        long embedding_idx = input[i];

        if(embedding_idx == padding_idx)
            continue;
        if(embedding_idx >= 0 && embedding_idx < num_embeddings)
        {
            if(embedding_idx == target_embedding_idx)
            {
                FLOAT scale = indices_freq
                                  ? (static_cast<FLOAT>(1.0f) / static_cast<FLOAT>(indices_freq[i]))
                                  : 1.0f;
                weight_grad_sum += output_grad[embedding_dim * i + j] * scale;
            }
        }
        else
        {
            if(index_error)
                index_error[0] = true;
        }
    }
    atomic_add_g(&weight_grad[target_embedding_idx * embedding_dim + j], weight_grad_sum);
}
