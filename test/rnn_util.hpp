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

#define MIO_RNN_TEST_EXPAVGFACTOR 0.1
#define MIO_RNN_TEST_EPSILON 1e-5 // FLT_EPSILON
#define MIO_RNN_SP_TEST_DEBUG 0



std::vector<int> get_rnn_num_layers()
{
    return {{1,2,4,6,7,8,10,20}};
}
        
std::vector<int> get_rnn_batchSize()
{
    return {{16,32,64,128}};
}


std::vector<int> get_rnn_seq_len()
{
    return {{1,2,4,10,20,50}};
}

std::vector<int> get_rnn_vector_len()
{
    return {{1,2,4,10,20,50}};
}

std::vector<int> get_rnn_hidden_size()
{
    return {{10,20,50,128}};
}


std::vector<int> generate_batchSeq()
{
    return {{11,10,9,8,5}};
}



std::vector<int> generate_batchSeq(const int batchSize, const int seqLength)
{
 
    int modval = 5;
    int currentval = batchSize;
    std::vector<int> batchseq;
    for(int i = 0; i < seqLength; i++)
    {
        printf("adding a value to batch sequence.\n");
        int nvalue = currentval - rand()%modval;
        currentval = (nvalue<1) ? 1 : nvalue;
        printf("current value: %d\n", currentval);
        batchseq.push_back(currentval);
    }
    return batchseq;
}






int sumvc(std::vector<int>& x)
{
    int sum = 0;
    for(int i = 0; i < x.size(); i++)
    {
        sum += x[i];
    }
    return sum;
}

float activfunc(float x, int actvf)
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

float dervactivfunc(float x, int actvf)
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







#define RNN_MM_TRANSPOSE 1
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

    size_t inner_loop = (!(a_flags & RNN_MM_TRANSPOSE)) ? a_cols : a_rows;

    if(!(a_flags & RNN_MM_TRANSPOSE) && !(b_flags & RNN_MM_TRANSPOSE))
    {
        for(size_t n = 0; n < c_rows; ++n)
        {
            for(size_t k = 0; k < c_cols; ++k)
            {
                Dtype mm_e = 0;
                for(size_t m = 0; m < inner_loop; ++m)
                {
                    mm_e += a_ptr[n * a_stride + m] * b_ptr[m * b_stride + k];
                }
                c_ptr[n * c_stride + k] = beta * c_ptr[n * c_stride + k] + alpha * mm_e;
            }
        }
    }
    else if((a_flags & RNN_MM_TRANSPOSE) && !(b_flags & RNN_MM_TRANSPOSE))
    {
        for(size_t n = 0; n < c_rows; ++n)
        {
            for(size_t k = 0; k < c_cols; ++k)
            {

                Dtype mm_e = 0;
                for(size_t m = 0; m < inner_loop; ++m)
                {
                    mm_e += a_ptr[m * a_stride + n] * b_ptr[m * b_stride + k];
#if 0
					if (
						(n == 0 && k == 33
						|| n == 1 && k == 32
						|| n == 3 && k == 1
						|| n == 4 && k == 0

						)
						&& a_ptr[m*a_stride + n] * b_ptr[m*b_stride + k] != 0
						)
					{
						printf("C:mm:%d %d %d   %11.9f %11.9f %11.9f %11.9f\n",
							n, k, m,
							mm_e, a_ptr[m*a_stride + n], b_ptr[m*b_stride + k], a_ptr[m*a_stride + n] * b_ptr[m*b_stride + k]);
					}
#endif
                }
                c_ptr[n * c_stride + k] = beta * c_ptr[n * c_stride + k] + alpha * mm_e;
            }
        }
    }
    else if(!(a_flags & RNN_MM_TRANSPOSE) && (b_flags & RNN_MM_TRANSPOSE))
    {
        for(size_t n = 0; n < c_rows; ++n)
        {
            for(size_t k = 0; k < c_cols; ++k)
            {
                Dtype mm_e = 0;

                for(size_t m = 0; m < inner_loop; ++m)
                {
                    mm_e += a_ptr[n * a_stride + m] * b_ptr[k * b_stride + m];
#if 0
					if (n == 0 && k == 6 && a_ptr[n*a_stride + m] * b_ptr[k*b_stride + m] != 0)
					{
						printf("%4d  %11.9f %11.9f %11.9f\n", m, mm_e, a_ptr[n*a_stride + m], b_ptr[k*b_stride + m]);
					}
#endif
                }
                c_ptr[n * c_stride + k] = beta * c_ptr[n * c_stride + k] + alpha * mm_e;
            }
        }
    }
    else
    {
        for(size_t n = 0; n < c_rows; ++n)
        {
            for(size_t k = 0; k < c_cols; ++k)
            {
                Dtype mm_e = 0;
                for(size_t m = 0; m < inner_loop; ++m)
                {
                    c_ptr[n * c_stride + k] += a_ptr[m * a_stride + n] * b_ptr[k * b_stride + m];
                }
                c_ptr[n * c_stride + k] = beta * c_ptr[n * c_stride + k] + alpha * mm_e;
            }
        }
    }
}


#endif


