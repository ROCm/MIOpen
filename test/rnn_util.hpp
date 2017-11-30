/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_RNN_UTIL_H_
#define MIOPEN_RNN_UTIL_H_

#include <cfloat>
#include <cmath>
#include <initializer_list>
#include <set>
#include <vector>
#include <cstdlib>

#include "gemm.hpp"

#define RNN_MM_TRANSPOSE 1

// RNN VANILLA configs
inline std::vector<int> get_rnn_num_layers()
{
    // return {{1, 5, 20}};
    return {{1, 5}};
}

inline std::vector<int> get_rnn_batchSize()
{
    // return {128};
    return {31};
}

inline std::vector<int> get_rnn_seq_len()
{
    // return {50};
    return {{3, 51}};
}

inline std::vector<int> get_rnn_vector_len()
{
    // return {32};
    return {31};
}

inline std::vector<int> get_rnn_hidden_size()
{
    // return {{16,64,128,256,1760,2048,2560}};
    return {127};
}

// LSTM configs
inline std::vector<int> get_lstm_num_layers() { return {{1, 5}}; }

inline std::vector<int> get_lstm_batchSize()
{
    // return {16};
    return {53};
}

inline std::vector<int> get_lstm_seq_len()
{
    return {25};
    // return {{2, 50}};
}

inline std::vector<int> get_lstm_vector_len()
{
    return {17};
    // return {{4, 32}};
}

inline std::vector<int> get_lstm_hidden_size()
{
    return {67};
    // return {{16,64,128,256,1760,2048,2560}};
}

// GRU configs
inline std::vector<int> get_gru_num_layers() { return {{1, 5}}; }

inline std::vector<int> get_gru_batchSize()
{
    // return {16};
    return {53};
}

inline std::vector<int> get_gru_seq_len()
{
    return {23};
    // return {{2, 50}};
}

inline std::vector<int> get_gru_vector_len()
{
    return {13};
    // return {{4, 32}};
}

inline std::vector<int> get_gru_hidden_size()
{
    return {67};
    // return {{16,64,128,256,1760,2048,2560}};
}

inline std::vector<std::vector<int>> generate_batchSeq(const int batchSize, const int seqLength)
{

    int modval = 3;
    srand(modval);
    int currentval = batchSize;
    std::vector<int> batchSeq;
    for(int i = 0; i < seqLength; i++)
    {
        if(i > 0)
        {
            int nvalue = currentval - rand() % modval;
            currentval = (nvalue < 1) ? 1 : nvalue;
            // printf("current value: %d\n", currentval);
        }
        // printf("adding a value to batch sequence: %d\n", currentval);
        batchSeq.push_back(currentval);
    }
    return {batchSeq};
}

inline int sumvc(std::vector<int>& x)
{
    int sum = 0;
    for(int i = 0; i < x.size(); i++)
    {
        sum += x[i];
    }
    return sum;
}

inline float activfunc(float x, int actvf)
{
    float alpha = 1, beta0 = 0, beta1 = 1;
    if(actvf == 0)
    {
        //        float y = 0;
        //        return std::max(x, y);
        return (x > 0) ? x : x * beta0;
    }
    else if(actvf == 2)
    {
        return 1 / (1 + exp(-x));
    }

    //    return tanh(x);
    return alpha * tanh(beta1 * x);
}

inline float dervactivfunc(float x, int actvf)
{
    if(actvf == 0)
    {
        return (x > 0 ? 1 : 0);
    }
    else if(actvf == 2)
    {
        return exp(-x) / (1 + exp(-x)) / (1 + exp(-x));
    }

    return 1 / cosh(x) / cosh(x);
}

template <typename Dtype>
void RNN_mm_cpu(const Dtype* a_ptr,
                size_t a_cols,
                size_t a_rows,
                size_t a_stride,
                int a_flags,
                const Dtype* b_ptr,
                size_t b_cols,
                size_t b_rows,
                size_t b_stride,
                int b_flags,
                Dtype* c_ptr,
                size_t c_cols,
                size_t c_rows,
                size_t c_stride,
                int /*c_flags*/,
                double d_alpha,
                double d_beta)
{

    Dtype alpha = Dtype(d_alpha);
    Dtype beta  = Dtype(d_beta);
    if((!(a_flags & RNN_MM_TRANSPOSE) && !(b_flags & RNN_MM_TRANSPOSE) &&
        ((a_cols != b_rows) || (a_rows != c_rows) || (b_cols != c_cols))) ||
       ((a_flags & RNN_MM_TRANSPOSE) && (b_flags & RNN_MM_TRANSPOSE) &&
        ((a_rows != b_cols) || (a_cols != c_rows) || (b_rows != c_cols))) ||
       ((a_flags & RNN_MM_TRANSPOSE) && !(b_flags & RNN_MM_TRANSPOSE) &&
        ((a_rows != b_rows) || (a_cols != c_rows) || (b_cols != c_cols))) ||
       (!(a_flags & RNN_MM_TRANSPOSE) && (b_flags & RNN_MM_TRANSPOSE) &&
        ((a_cols != b_cols) || (a_rows != c_rows) || (b_rows != c_cols))))
    {
        printf("MM_CPU ERROR; %zd %zd   %zd %zd   %zd %zd\n",
               a_cols,
               a_rows,
               b_cols,
               b_rows,
               c_rows,
               c_cols);
        return;
    }

    auto c_out = [&](int i, int j, double x) {
        c_ptr[i * c_stride + j] = beta * c_ptr[i * c_stride + j] + alpha * x;
    };

    size_t inner_loop = (!(a_flags & RNN_MM_TRANSPOSE)) ? a_cols : a_rows;

    if(!(a_flags & RNN_MM_TRANSPOSE) && !(b_flags & RNN_MM_TRANSPOSE))
    {
        gemm(c_rows,
             c_cols,
             inner_loop,
             with_stride(a_ptr, a_stride),
             with_stride(b_ptr, b_stride),
             c_out);
    }
    else if((a_flags & RNN_MM_TRANSPOSE) && !(b_flags & RNN_MM_TRANSPOSE))
    {
        gemm(c_rows,
             c_cols,
             inner_loop,
             miopen::flip(with_stride(a_ptr, a_stride)),
             with_stride(b_ptr, b_stride),
             c_out);
    }
    else if(!(a_flags & RNN_MM_TRANSPOSE) && (b_flags & RNN_MM_TRANSPOSE))
    {
        gemm(c_rows,
             c_cols,
             inner_loop,
             with_stride(a_ptr, a_stride),
             miopen::flip(with_stride(b_ptr, b_stride)),
             c_out);
    }
    else
    {
        gemm(c_rows,
             c_cols,
             inner_loop,
             miopen::flip(with_stride(a_ptr, a_stride)),
             miopen::flip(with_stride(b_ptr, b_stride)),
             c_out);
    }
}

#endif
