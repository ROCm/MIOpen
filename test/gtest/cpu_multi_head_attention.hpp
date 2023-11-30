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
tensor<T> concatenateHeads(std::vector<tensor<T>>& heads, size_t batch_size, 
    size_t seq_length, size_t problem_dim, size_t d_k)
{
    std::vector<size_t> multi_head_lens = {batch_size, seq_length, problem_dim};
    tensor<T> multi_head({multi_head_lens});

    for(size_t b = 0; b < batch_size; ++b)
    {
        for(size_t s = 0; s < seq_length; ++s)
        {
            size_t dim = 0;

            for(const auto&each_head : heads)
            {
                for(size_t d = 0; d < d_k; ++d,++dim)
                {
                    multi_head(b, s, dim) = each_head(b, s, d);
                }
            }
        }
    }
    return multi_head;

}

template <class T>
void softmax(tensor<T>& tensor_val)
{
    // assert(tensor_val.desc.GetLengths() == static_cast<size_t>(2));
    size_t batch = tensor_val.desc.GetLengths()[0];
    size_t rows = tensor_val.desc.GetLengths()[1];
    size_t cols = tensor_val.desc.GetLengths()[2];
    for(int b = 0; b < batch; ++b)
    {
        for(int i = 0; i < rows; ++i)
        {
            // T max_score_in_row = *std::max_element(tensor_val.begin(), tensor_val.begin() + cols); 
            // computing sum of exp for each row
            T sum(0);
            for(int j = 0; j < cols; ++j)
            {
                tensor_val(b, i, j) = std::exp(tensor_val(b, i, j));
                sum += tensor_val(b, i, j);
            }

            // applying the softmax
            for(int j = 0; j < cols; ++j)
            {
                tensor_val(b, i, j) = tensor_val(b, i,j)/sum;
            }
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
void print(const tensor<T>& tensor_val,
           size_t problem_dimension)
{
    size_t count = 0;
    for(const auto& val : tensor_val)
    {
        if(count%problem_dimension == 0)
            std::cout << "\n";
        if(count == tensor_val.data.size()/2)
        {
            std::cout << "\n\n";
        }
        std::cout << std::fixed << std::setprecision(2) << val << " , ";
        count++;
    }
    std::cout << "\n======\n";

}
template<typename T>
tensor<T> multi_head_attention(size_t num_heads,
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
    std::mutex coutMutex;


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
        set_of_k_transpose[head].par_for_each([&](size_t b_id, size_t dk_id, size_t sl_id)
        {
            set_of_k_transpose[head](b_id, dk_id, sl_id) = set_of_k[head](b_id, sl_id, dk_id);
        });
    }
    // print(set_of_k_transpose[0], problem_dimension);


    double sqrt_dk = squareRoot(d_k);
    for(size_t head = 0; head < num_heads; ++head){
        set_of_q_dot_k_transpose[head].par_for_each(
            [&](size_t b_id, size_t sl_r_id, size_t sl_c_id){
                T sum(0);
                for(size_t dk_id = 0; dk_id < d_k; ++dk_id)
                {
                    sum += set_of_q[head](b_id, sl_r_id, dk_id) * set_of_k_transpose[head](b_id, dk_id, sl_c_id);
                }
                sum = (sum/sqrt_dk) + mask_val(sl_r_id, sl_c_id);
                set_of_q_dot_k_transpose[head](b_id, sl_r_id, sl_c_id) = sum;
            });
    }

    for(size_t head = 0; head < num_heads; ++head){
        // std::cout << "compute softmax of " << head << std::endl;
        softmax(set_of_q_dot_k_transpose[head]);

        // dropout
        tensor<T> rand_dis(set_of_q_dot_k_transpose[head].desc.GetLengths());
        rand_dis.generate(GenData<T>{});
        set_of_q_dot_k_transpose[head].par_for_each( 
            [&](size_t b_id, size_t sl_r_id, size_t sl_c_id) { 
                if(rand_dis(b_id, sl_r_id, sl_c_id) < drop_out_rate){
                    set_of_q_dot_k_transpose[head](b_id, sl_r_id, sl_c_id) 
                        = T(0);
                }
            });
    }

    for(size_t head = 0; head < num_heads; ++head){
        atten_heads[head].par_for_each(
            [&](size_t b_id, size_t s_id, size_t p_id){
                T sum(0);
                for(size_t sl_id = 0; sl_id < sequence_length; ++sl_id)
                {
                    sum += set_of_q_dot_k_transpose[head](b_id, s_id, sl_id) * set_of_v[head](b_id, p_id, sl_id);
                }
                atten_heads[head](b_id, s_id, p_id) = sum;
            });
        // print(atten_heads[head], d_k);
    }


    // concatinate attention heads 
    tensor<T> multi_head = concatenateHeads(atten_heads,
                            batch_size, sequence_length, problem_dimension, d_k);

    tensor<T> linear_multi_head(multi_head.desc.GetLengths());

    for(size_t head = 0; head < num_heads; ++head){
        multi_head.par_for_each(
            [&](size_t b_id, size_t s_id, size_t p_id){
                std::lock_guard<std::mutex> guard(coutMutex);

                T sum(0);
                for(size_t pd_id = 0; pd_id < problem_dimension; ++pd_id)
                {
                    sum += multi_head(b_id, s_id, pd_id) * 
                           set_of_head_weights[head](p_id, pd_id);
                }
                linear_multi_head(b_id, s_id, p_id) = sum;
            });
    }

    return linear_multi_head;
}

}
}
