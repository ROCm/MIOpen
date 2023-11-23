/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#pragma once

#include "tensor_holder.hpp"
#include "conv_tensor_gen.hpp"
#include <type_traits>

namespace test {
namespace cpu {

template <typename T>
auto squareRoot(T value) -> decltype(std::sqrt(value)) {
    static_assert(std::is_arithmetic<T>::value, "squareRoot only supports arithmetic types");
    return std::sqrt(value);
}

template <class T>
void softmax(tensor<T>& tensor_val)
{
    // assert(tensor_val.desc.GetLengths() == static_cast<size_t>(2));
    size_t rows = tensor_val.desc.GetLengths()[0];
    size_t cols = tensor_val.desc.GetLengths()[1];

    for(int i = 0; i < rows; ++i)
    {
        T max_score_in_row = *std::max_element(tensor_val.begin(), tensor_val.begin() + cols); 
        // computing sum of exp for each row
        T sum(0);
        for(int j = 0; j < cols; ++j)
        {
            sum += std::exp(tensor_val(i, j) - max_score_in_row);
        }

        // applying the softmax
        for(int j = 0; j < cols; ++j)
        {
            tensor_val(i,j) = std::exp(tensor_val(i,j) - max_score_in_row) / sum;
        }
    }
}

template<typename T>
void par_compute_val(size_t num_heads, size_t problem_dimension, std::vector<tensor<T>> set_of_weights,
                    tensor<T> word_position, std::vector<tensor<T>>& ret_value)
{
    for(size_t head = 0; head < num_heads; ++head){
        ret_value[head].par_for_each(
            [&](size_t b_id, size_t s_id, size_t dk_id){
                T sum(0);
                for(size_t pd_id = 0; pd_id < problem_dimension; ++pd_id)
                {
                    // quick change is pd_id and dk_id
                    sum += word_position(b_id, s_id, pd_id) * set_of_weights[head](pd_id, dk_id);
                }
                ret_value[head](b_id, s_id, dk_id) = sum;
            });
    }
}

template<typename T>
void multi_head_attention(size_t num_heads,
                          size_t problem_dimension,
                          size_t sequence_length,
                          size_t batch_size, 
                          tensor<T> mask_val,
                          float drop_out_rate)
{
    size_t d_k = problem_dimension/num_heads;
    // In each head of the Transformer's multi-head attention mechanism, the same weight matrix 
    // is applied across all batches.
    std::vector<size_t> qkv_weights_lens = {problem_dimension, d_k};
    std::vector<tensor<T>> set_of_q_weights(num_heads, tensor<T>{qkv_weights_lens});
    std::vector<tensor<T>> set_of_k_weights(num_heads, tensor<T>{qkv_weights_lens});
    std::vector<tensor<T>> set_of_v_weights(num_heads, tensor<T>{qkv_weights_lens});

    std::vector<size_t> qkv_lens = {batch_size, sequence_length, d_k};
    std::vector<tensor<T>> set_of_q(num_heads, tensor<T>{qkv_lens});
    std::vector<tensor<T>> set_of_k(num_heads, tensor<T>{qkv_lens});
    std::vector<tensor<T>> set_of_v(num_heads, tensor<T>{qkv_lens});
    std::vector<tensor<T>> atten_heads(num_heads, tensor<T>{qkv_lens});

    std::vector<size_t> k_trans_lens = {batch_size, d_k, sequence_length};
    std::vector<tensor<T>> set_of_k_transpose(num_heads, tensor<T>{k_trans_lens});
    std::vector<size_t> q_dot_k_trans_lens = {batch_size, sequence_length, sequence_length};
    std::vector<tensor<T>> set_of_q_dot_k_transpose(num_heads, tensor<T>{q_dot_k_trans_lens});

    std::vector<size_t> multi_head_weights_lens = {problem_dimension, problem_dimension};
    std::vector<tensor<T>> set_of_head_weights(num_heads, tensor<T>{multi_head_weights_lens});
    
    std::vector<size_t> lens = {batch_size, sequence_length, problem_dimension};
    tensor<T> word_position({lens});
    // final output stored here
    tensor<T> concatinated_multi_head_attention{lens};

    word_position.generate(GenData<T>{});

    for(int i = 0; i < num_heads; ++i)
    {
        set_of_q_weights[i].generate(GenData<T>{});
        set_of_k_weights[i].generate(GenData<T>{});
        set_of_v_weights[i].generate(GenData<T>{});
        set_of_head_weights[i].generate(GenData<T>{});
    }
    
    par_compute_val(num_heads, problem_dimension, set_of_q_weights, word_position, set_of_q);
    par_compute_val(num_heads, problem_dimension, set_of_k_weights, word_position, set_of_k);
    par_compute_val(num_heads, problem_dimension, set_of_v_weights, word_position, set_of_v);
    
    for(size_t head = 0; head < num_heads; ++head){
        set_of_k_transpose[head].par_for_each([&](size_t b, size_t s, size_t d)
        {
            set_of_k_transpose[head](b, d, s) = set_of_k[head](b, s, d);
        });
    }

    for(size_t head = 0; head < num_heads; ++head){
        set_of_q_dot_k_transpose[head].par_for_each(
            [&](size_t b_id, size_t s_r_id, size_t s_c_id){
                T sum(0);
                for(size_t dk_id = 0; dk_id < d_k; ++dk_id)
                {
                    // quick change is pd_id and dk_id
                    sum += set_of_q[head](b_id, s_r_id, dk_id) * set_of_k_transpose[head](dk_id, s_c_id);
                }
                set_of_q_dot_k_transpose[head](b_id, s_r_id, s_c_id) = sum;
            });
    }

    double sqrt_dk = squareRoot(d_k);
    for(size_t head = 0; head < num_heads; ++head){
        set_of_q_dot_k_transpose[head].par_for_each(
            [&](int i, int j) { 
                set_of_q_dot_k_transpose[head](i, j) = set_of_q_dot_k_transpose[head](i, j)/sqrt_dk;
            });
    }

    par_compute_val(num_heads, problem_dimension, set_of_v_weights, word_position, set_of_v);
    

    // masking
    for(size_t head = 0; head < num_heads; ++head){
        set_of_q_dot_k_transpose[head].par_for_each(
            [&](int i, int j) { 
                set_of_q_dot_k_transpose[head](i, j) += mask_val(i, j);
        });
    }

    for(size_t head = 0; head < num_heads; ++head){
        softmax(set_of_q_dot_k_transpose[head]);
        // dropout
        tensor<T> rand_dis(set_of_q_dot_k_transpose[head].desc.GetLengths());
        rand_dis.generate(GenData<T>{});
        set_of_q_dot_k_transpose[head].par_for_each( 
            [&](int i, int j) { if(rand_dis(i,j) < drop_out_rate){
                set_of_q_dot_k_transpose[head](i,j) = T(0);
            }});
    }

    for(size_t head = 0; head < num_heads; ++head){
        atten_heads[head].par_for_each(
            [&](size_t b_id, size_t s_r_id, size_t s_c_id){
                T sum(0);
                for(size_t dk_id = 0; dk_id < d_k; ++dk_id)
                {
                    sum += set_of_q[head](b_id, s_r_id, dk_id) * set_of_v[head](dk_id, s_c_id);
                }
                atten_heads[head](b_id, s_r_id, s_c_id) = sum;
            });
    }

    // concatinate single_attn 
    std::vector<T> concatinated_data;

    for(size_t head = 0; head < num_heads; ++head)
    {
        concatinated_data.insert(concatinated_data.end(),
                                set_of_head_weights[head].data.begin(),
                                set_of_head_weights[head].data.end());
    }
    std::vector<size_t> multi_head_lens = {batch_size, sequence_length, problem_dimension};
    tensor<T> multi_head({multi_head_lens});
    multi_head.data = concatinated_data;

    // multi_head dot with multi_head_weights

}

}
}
