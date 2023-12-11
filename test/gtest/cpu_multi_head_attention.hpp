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

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/layernorm.hpp>

#include "tensor_holder.hpp"
#include "cpu_layernorm.hpp"
#include "get_handle.hpp"
#include "../driver/tensor_driver.hpp"
#include "verify.hpp"
#include <random>

namespace test {
namespace cpu {

template <typename T>
auto squareRoot(T value) -> decltype(std::sqrt(value))
{
    static_assert(std::is_arithmetic<T>::value, "squareRoot only supports arithmetic types");
    return std::sqrt(value);
}

template <typename T>
void BrodcastParMul(tensor<T> A_mat, tensor<T> B_mat, tensor<T>& ret_value)
{
    // std::mutex coutMutex;
    size_t A_col = A_mat.desc.GetLengths()[3];
    ret_value.par_for_each([&](size_t b_id, size_t h_id, size_t s_id, size_t d_id) {
        T sum(0);

        for(size_t pd_id = 0; pd_id < A_col; ++pd_id)
        {
            // std::lock_guard<std::mutex> guard(coutMutex);
            sum += A_mat(b_id, 0, s_id, pd_id) * B_mat(0, h_id, pd_id, d_id);
        }
        ret_value(b_id, h_id, s_id, d_id) = sum;
    });
}

template <typename T>
void ParMul(tensor<T> A_mat, tensor<T> B_mat, tensor<T>& ret_value, size_t key_dim = 1)
{
    std::mutex coutMutex;
    size_t A_col = A_mat.desc.GetLengths()[3];
    ret_value.par_for_each([&](size_t b_id, size_t h_id, size_t s_id, size_t d_id) {
        T sum(0);
        std::lock_guard<std::mutex> guard(coutMutex);
        for(size_t pd_id = 0; pd_id < A_col; ++pd_id)
        {
            sum += A_mat(b_id, h_id, s_id, pd_id) * B_mat(b_id, h_id, pd_id, d_id);
        }
        ret_value(b_id, h_id, s_id, d_id) = sum;
    });
}

template <typename T>
void ParMulAttention(tensor<T> A_mat, tensor<T> B_mat, tensor<T>& ret_value, size_t d_k)
{
    std::mutex coutMutex;
    // sequence length
    size_t A_col = A_mat.desc.GetLengths()[3];
    ret_value.par_for_each([&](size_t b_id, size_t h_id, size_t s_id, size_t d_id) {
        T sum(0);
        size_t dk_id = d_id % d_k;
        std::lock_guard<std::mutex> guard(coutMutex);
        for(size_t pd_id = 0; pd_id < A_col; ++pd_id)
        {
            sum += A_mat(b_id, h_id, s_id, pd_id) * B_mat(b_id, h_id, pd_id, dk_id);
        }
        ret_value(b_id, h_id, s_id, d_id) = sum;
    });
}

template <typename T>
void print(const tensor<T>& tensor_val)
{
    std::cout << "\n=================start=====================\n";
    size_t batch_size = tensor_val.desc.GetLengths()[0];
    size_t head_size  = tensor_val.desc.GetLengths()[1];
    size_t row_size   = tensor_val.desc.GetLengths()[2];
    size_t col_size   = tensor_val.desc.GetLengths()[3];
    for(size_t b = 0; b < batch_size; ++b)
    {
        std::cout << "batch = " << b << std::endl;
        for(size_t h = 0; h < head_size; ++h)
        {
            std::cout << "head = " << h << std::endl;
            for(size_t r = 0; r < row_size; ++r)
            {
                for(size_t c = 0; c < col_size; ++c)
                {
                    std::cout << std::fixed << std::setprecision(2) << tensor_val(b, h, r, c)
                              << " , ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n=================end=====================\n";
}

template <typename T>
void SetupMask(tensor<T>& mask)
{
    // std::mutex coutMutex;
    mask.par_for_each([&](size_t b_id, size_t h_id, size_t s_id, size_t p_id) {
        // make anything above diagonal inf
        if(p_id > s_id)
        {
            // std::lock_guard<std::mutex> guard(coutMutex);
            // std::cout << s_id << "," << p_id << std::endl;
            mask(b_id, h_id, s_id, p_id) = -std::numeric_limits<T>::infinity();
        }
    });
}

template <typename T>
void MatTranspose(const tensor<T>& mat_val, tensor<T>& trans_mat)
{
    trans_mat.par_for_each([&](size_t b_id, size_t h_id, size_t dk_id, size_t sl_id) {
        trans_mat(b_id, h_id, dk_id, sl_id) = mat_val(b_id, h_id, sl_id, dk_id);
    });
}

template <typename T>
void Scale(tensor<T>& mat_val, double scale_factor)
{
    // assert scale_factor is != 0.0 .. tolerance
    mat_val.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        mat_val(b_id, h_id, sl_id, dk_id) = mat_val(b_id, h_id, sl_id, dk_id) * scale_factor;
    });
}

template <typename T>
void AddMask(tensor<T>& mat_A_val, const tensor<T>& mat_mask)
{
    mat_A_val.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        mat_A_val(b_id, h_id, sl_id, dk_id) =
            mat_A_val(b_id, h_id, sl_id, dk_id) + mat_mask(0, 0, sl_id, dk_id);
    });
}

template <typename T>
void Sub(tensor<T>& mat_A_val, const tensor<T>& mat_sub)
{
    mat_A_val.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        mat_A_val(b_id, h_id, sl_id, dk_id) =
            mat_A_val(b_id, h_id, sl_id, dk_id) - mat_sub(b_id, h_id, sl_id, 0);
    });
}

template <class T>
void RowReductionMax(tensor<T>& tensor_val, tensor<T>& m_tensor)
{
    size_t batch  = tensor_val.desc.GetLengths()[0];
    size_t head   = tensor_val.desc.GetLengths()[1];
    size_t sq_len = tensor_val.desc.GetLengths()[2];

    m_tensor = tensor<T>{batch, head, sq_len, 1};

    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < head; ++h)
        {
            T max_score_in_row;
            for(int sl = 0; sl < sq_len; ++sl)
            {
                size_t start_index =
                    b * (head * sq_len * sq_len) + h * (sq_len * sq_len) + sl * sq_len;
                size_t end_index   = start_index + sq_len;
                max_score_in_row   = *std::max_element(tensor_val.begin() + start_index,
                                                     tensor_val.begin() + end_index);
                m_tensor(b, h, sl) = max_score_in_row;
            }
        }
    }
}

template <class T>
void Zinv(tensor<T>& tensor_val, tensor<T>& zinv_tensor)
{
    size_t batch  = tensor_val.desc.GetLengths()[0];
    size_t head   = tensor_val.desc.GetLengths()[1];
    size_t sq_len = tensor_val.desc.GetLengths()[2];

    zinv_tensor = tensor<T>{batch, head, sq_len, 1};

    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < head; ++h)
        {
            T sum(0);
            for(int sl = 0; sl < sq_len; ++sl)
            {
                size_t start_index =
                    b * (head * sq_len * sq_len) + h * (sq_len * sq_len) + sl * sq_len;
                size_t end_index = start_index + sq_len;
                sum              = std::accumulate(
                    tensor_val.begin() + start_index, tensor_val.begin() + end_index, 0.0);
                zinv_tensor(b, h, sl) = 1 / sum;
            }
        }
    }
}

template <class T>
void Exponent(tensor<T>& tensor_val)
{
    tensor_val.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        tensor_val(b_id, h_id, sl_id, dk_id) = std::exp(tensor_val(b_id, h_id, sl_id, dk_id));
    });
}

template <class T>
void ZinvMultiply(tensor<T>& q_dot_k_transpose, tensor<T>& zinv_tensors)
{
    q_dot_k_transpose.par_for_each([&](size_t b_id, size_t h_id, size_t sl_r_id, size_t sl_c_id) {
        q_dot_k_transpose(b_id, h_id, sl_r_id, sl_c_id) =
            q_dot_k_transpose(b_id, h_id, sl_r_id, sl_c_id) / zinv_tensors(b_id, h_id, sl_r_id);
    });
}

template <class T>
void DropOut(tensor<T>& q_dot_k_transpose, const double& drop_out_rate)
{
    tensor<T> rand_dis(q_dot_k_transpose.desc.GetLengths());
    rand_dis.generate(GenData<T>{});
    q_dot_k_transpose.par_for_each([&](size_t b_id, size_t h_id, size_t sl_r_id, size_t sl_c_id) {
        if(rand_dis(b_id, h_id, sl_r_id, sl_c_id) < drop_out_rate)
        {
            q_dot_k_transpose(b_id, h_id, sl_r_id, sl_c_id) = T(0);
        }
    });
}

struct CPUMHATestCase
{
    size_t batch_size;
    size_t num_heads;
    size_t sequence_length;
    size_t problem_dimension;
    float drop_out_rate;
};
std::vector<CPUMHATestCase> CPUMHAConfigs() { return {{2, 2, 4, 4, 0.2}}; }

template <typename T = float>
struct CPUMHATest : public ::testing::TestWithParam<CPUMHATestCase>
{
protected:
    void SetUp() override
    {
        cpu_mha_test_case = GetParam();
        d_k               = cpu_mha_test_case.problem_dimension / cpu_mha_test_case.num_heads;
        init();
        // mult-head attention begin
        // 0) create Q, K and V
        BrodcastParMul(word_position, q_weights, q_val);
        BrodcastParMul(word_position, k_weights, k_val);
        BrodcastParMul(word_position, v_weights, v_val);

        // //  1) Reshape Transpose
        MatTranspose(k_val, k_transpose);
        // //  2) BMM
        ParMul(q_val, k_val, q_dot_k_transpose);

        // // 2.2) Adding mask
        AddMask(q_dot_k_transpose, mask);

        // //   3) Attention Scale
        double sqrt_dk = 1.0 / squareRoot(d_k);
        Scale(q_dot_k_transpose, sqrt_dk);

        double fp8_scale = 1.0 / 1.0;
        // //   4) descale Q
        Scale(q_val, fp8_scale);
        // //   5) descale K
        Scale(k_val, fp8_scale);
        // //   6) Row Reduction Max => M
        RowReductionMax(q_dot_k_transpose, rrm);

        // //   7) rrm substraction
        Sub(q_dot_k_transpose, rrm);
        // //   8) pointwise exponentiation
        Exponent(q_dot_k_transpose);

        // //   9) Zinv
        Zinv(q_dot_k_transpose, zinv_tensors);
        // //  10) Zinv reciprocal
        ZinvMultiply(q_dot_k_transpose, zinv_tensors);

        // //  11) drop out
        DropOut(q_dot_k_transpose, cpu_mha_test_case.drop_out_rate);

        // //  11.1) drop out scalse
        double drop_out_scale = 1.0 / (1.0 - d_k);
        Scale(q_dot_k_transpose, drop_out_scale);

        // //  12) descale Q
        double scale_s = 1.0;
        Scale(q_dot_k_transpose, scale_s);
        // //  13) O = A*V
        ParMulAttention(q_dot_k_transpose, v_val, atten_heads, d_k);
    }

    void TearDown() override
    {
        // verify
    }

private:
    void init()
    {
        mask =
            tensor<T>{1, 1, cpu_mha_test_case.sequence_length, cpu_mha_test_case.sequence_length};
        SetupMask(mask);

        q_val = tensor<T>{cpu_mha_test_case.batch_size,
                          cpu_mha_test_case.num_heads,
                          cpu_mha_test_case.sequence_length,
                          d_k};
        k_val = q_val;
        v_val = q_val;

        q_weights =
            tensor<T>{1, cpu_mha_test_case.num_heads, cpu_mha_test_case.problem_dimension, d_k};
        k_weights = q_weights;
        v_weights = q_weights;

        atten_heads = tensor<T>{cpu_mha_test_case.batch_size,
                                cpu_mha_test_case.num_heads,
                                cpu_mha_test_case.sequence_length,
                                cpu_mha_test_case.problem_dimension};

        k_transpose = tensor<T>{cpu_mha_test_case.batch_size,
                                cpu_mha_test_case.num_heads,
                                d_k,
                                cpu_mha_test_case.sequence_length};

        q_dot_k_transpose = tensor<T>{cpu_mha_test_case.batch_size,
                                      cpu_mha_test_case.num_heads,
                                      cpu_mha_test_case.sequence_length,
                                      cpu_mha_test_case.sequence_length};

        word_position = tensor<T>{cpu_mha_test_case.batch_size,
                                  1,
                                  cpu_mha_test_case.sequence_length,
                                  cpu_mha_test_case.problem_dimension};

        word_position.generate(GenData<T>{});
        q_weights.generate(GenData<T>{});
        k_weights.generate(GenData<T>{});
        v_weights.generate(GenData<T>{});
    }

    tensor<T> q_weights;
    tensor<T> k_weights;
    tensor<T> v_weights;

    tensor<T> q_val;
    tensor<T> k_val;
    tensor<T> v_val;

    tensor<T> k_transpose;
    tensor<T> q_dot_k_transpose;

    tensor<T> atten_heads;

    tensor<T> word_position;

    std::mutex coutMutex;
    tensor<T> mask;

    CPUMHATestCase cpu_mha_test_case;
    tensor<T> mha;
    // row reduction max
    tensor<T> rrm;
    tensor<T> zinv_tensors;
    size_t d_k;
};

} // namespace cpu
} // namespace test
