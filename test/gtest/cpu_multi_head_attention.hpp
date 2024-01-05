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
#include <hip_float8.hpp>

using float8 = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;

namespace test {
namespace cpu {

template <typename T>
void print5(const tensor<T>& tensor_val, std::string header_msg = "start")
{
    std::cout << "\n================= " << header_msg << " =====================\n";
    for(auto it : tensor_val.desc.GetLengths())
    {
        std::cout << it << ",";
    }
    std::cout << "\n\n";
    size_t i_size = tensor_val.desc.GetLengths()[0];
    size_t j_size = tensor_val.desc.GetLengths()[1];
    size_t k_size = tensor_val.desc.GetLengths()[2];
    size_t l_size = tensor_val.desc.GetLengths()[3];
    size_t m_size = tensor_val.desc.GetLengths()[4];
    for(size_t i = 0; i < i_size; ++i)
    {
        for(size_t j = 0; j < j_size; ++j)
        {
            for(size_t k = 0; k < k_size; ++k)
            {
                for(size_t l = 0; l < l_size; ++l)
                {
                    for(size_t m = 0; m < m_size; ++m)
                    {
                        std::cout << std::fixed << std::setprecision(2) << tensor_val(i, j, k, l, m)
                                  << " , ";
                    }
                    std::cout << "\n";
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
void print4(const tensor<T>& tensor_val, std::string header_msg = "start")
{
    std::cout << "\n================= " << header_msg << " =====================\n";
    size_t i_size = tensor_val.desc.GetLengths()[0];
    size_t j_size = tensor_val.desc.GetLengths()[1];
    size_t k_size = tensor_val.desc.GetLengths()[2];
    size_t l_size = tensor_val.desc.GetLengths()[3];
    std::cout << i_size << "," << j_size << ", " << k_size << "," << l_size << std::endl;
    for(size_t i = 0; i < i_size; ++i)
    {
        for(size_t j = 0; j < j_size; ++j)
        {
            for(size_t k = 0; k < k_size; ++k)
            {
                for(size_t l = 0; l < l_size; ++l)
                {
                    std::cout << std::fixed << std::setprecision(2) << tensor_val(i, j, k, l)
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
void print3(const tensor<T>& tensor_val, std::string header_msg = "start")
{
    std::cout << "\n================= " << header_msg << " =====================\n";
    size_t i_size = tensor_val.desc.GetLengths()[0];
    size_t j_size = tensor_val.desc.GetLengths()[1];
    size_t k_size = tensor_val.desc.GetLengths()[2];
    std::cout << i_size << "," << j_size << ", " << k_size << std::endl;
    for(size_t i = 0; i < i_size; ++i)
    {
        for(size_t j = 0; j < j_size; ++j)
        {
            for(size_t k = 0; k < k_size; ++k)
            {
                std::cout << std::fixed << std::setprecision(2) << tensor_val(i, j, k) << " , ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n=================end=====================\n";
}

// C_mat = A_mat.dot(B_mat)
// A_mat : 4D
// B_mat : 3D
// C_mat : 5D
template <typename T>
void Dot_4D_3D(const tensor<T>& A_mat, const tensor<T>& B_mat, tensor<T>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[3];
    assert(k_val == B_mat.desc.GetLengths()[1]);
    C_mat.par_for_each([&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += A_mat(b_id, sc_id, sl_id, k_id) * B_mat(h_id, k_id, dk_id);
        }
        C_mat(b_id, sc_id, h_id, sl_id, dk_id) = sum;
    });
}

template <typename T>
void Dot_5D_3D(const tensor<T>& A_mat, const tensor<T>& B_mat, tensor<T>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[4];
    // assert(k_val == B_mat.desc.GetLengths()[1]);
    C_mat.par_for_each([&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += A_mat(b_id, sc_id, h_id, sl_id, k_id) * B_mat(h_id, k_id, dk_id);
        }
        C_mat(b_id, sc_id, h_id, sl_id, dk_id) = sum;
    });
}

// C_mat = A_mat.dot(B_mat)
// A_mat : 5D
// B_mat : 5D
// C_mat : 5D
template <typename T>
void Dot_5D_5D(const tensor<T>& A_mat, const tensor<T>& B_mat, tensor<T>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[4];
    assert(k_val == B_mat.desc.GetLengths()[3]);
    C_mat.par_for_each([&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += A_mat(b_id, sc_id, h_id, sl_id, k_id) * B_mat(b_id, sc_id, h_id, k_id, dk_id);
        }

        C_mat(b_id, sc_id, h_id, sl_id, dk_id) = sum;
    });
}

template <typename T>
void AddMask5D_2D(tensor<T>& mat_A_val, const tensor<T>& mat_mask)
{
    mat_A_val.par_for_each(
        [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            mat_A_val(b_id, sc_id, h_id, sl_i_id, sl_j_id) =
                mat_A_val(b_id, sc_id, h_id, sl_i_id, sl_j_id) + mat_mask(sl_i_id, sl_j_id);
        });
}

// Computes the sum of the row in A_mat
// A_mat : 5D
// rrm_tensor : 5D
template <class T>
void RowReductionMax(const tensor<T>& A_mat, tensor<T>& rrm_tensor)
{
    size_t sl_dim = A_mat.desc.GetLengths()[3];
    rrm_tensor.par_for_each(
        [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_id, size_t sl0_id) {
            T max(A_mat(b_id, sc_id, h_id, sl_id, sl0_id));
            for(size_t id = 0; id < sl_dim; ++id)
            {
                if(A_mat(b_id, sc_id, h_id, sl_id, id) > max)
                {
                    max = A_mat(b_id, sc_id, h_id, sl_id, id);
                }
            }
            rrm_tensor(b_id, sc_id, h_id, sl_id, sl0_id) = max;
        });
}

template <class T>
void Zinv(const tensor<T>& A_mat, tensor<T>& zinv_tensor)
{
    size_t sl_dim = A_mat.desc.GetLengths()[3];
    T one(1);
    zinv_tensor.par_for_each(
        [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_id, size_t sl0_id) {
            double sum(0);
            for(size_t id = 0; id < sl_dim; ++id)
            {
                sum += A_mat(b_id, sc_id, h_id, sl_id, id);
            }
            zinv_tensor(b_id, sc_id, h_id, sl_id, sl0_id) = one / sum;
        });
}

template <typename T>
void Scale(tensor<T>& mat_val, double scale_factor)
{
    // assert scale_factor is != 0.0 .. tolerance
    mat_val.par_for_each(
        [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            mat_val(b_id, sc_id, h_id, sl_i_id, sl_j_id) =
                mat_val(b_id, sc_id, h_id, sl_i_id, sl_j_id) * scale_factor;
        });
}

template <typename T>
void Sub(tensor<T>& mat_A_val, const tensor<T>& mat_sub)
{
    mat_A_val.par_for_each(
        [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            mat_A_val(b_id, sc_id, h_id, sl_i_id, sl_j_id) =
                mat_A_val(b_id, sc_id, h_id, sl_i_id, sl_j_id) -
                mat_sub(b_id, sc_id, h_id, sl_i_id, 0);
        });
}

template <class T>
void Exponent(tensor<T>& tensor_val)
{
    tensor_val.par_for_each(
        [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            tensor_val(b_id, sc_id, h_id, sl_i_id, sl_j_id) =
                std::exp(tensor_val(b_id, sc_id, h_id, sl_i_id, sl_j_id));
        });
}

template <class T>
void ZinvMultiply(tensor<T>& q_dot_k_transpose, const tensor<T>& zinv_tensors)
{
    q_dot_k_transpose.par_for_each(
        [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            q_dot_k_transpose(b_id, sc_id, h_id, sl_i_id, sl_j_id) =
                q_dot_k_transpose(b_id, sc_id, h_id, sl_i_id, sl_j_id) *
                zinv_tensors(b_id, sc_id, h_id, sl_i_id, 0);
        });
}

template <typename T>
void SetupMask(tensor<T>& mask)
{
    mask.par_for_each([&](size_t s_id, size_t p_id) {
        // make anything above diagonal inf
        if(p_id > s_id)
        {
            mask(s_id, p_id) = -std::numeric_limits<T>::infinity();
        }
    });
}

template <typename T>
void Transpose_5D(const tensor<T>& mat_val, tensor<T>& trans_mat)
{
    trans_mat.par_for_each([&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_id, size_t dk_id) {
        trans_mat(b_id, sc_id, h_id, sl_id, dk_id) = mat_val(b_id, sc_id, h_id, dk_id, sl_id);
    });
}

template <class T>
void DropOut(tensor<T>& q_dot_k_transpose, const double& drop_out_rate)
{
    tensor<T> rand_dis(q_dot_k_transpose.desc.GetLengths());
    rand_dis.generate(GenData<T>{});
    q_dot_k_transpose.par_for_each(
        [&](size_t b_id, size_t sc_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            if(rand_dis(b_id, sc_id, h_id, sl_i_id, sl_j_id) < drop_out_rate)
            {
                q_dot_k_transpose(b_id, sc_id, h_id, sl_i_id, sl_j_id) = T(0);
            }
        });
}

struct CPUMHATestCase
{
    size_t batch_size;
    size_t sequence_count;
    size_t num_heads;
    size_t sequence_length;
    size_t problem_dimension;
    float drop_out_rate;

    friend std::ostream& operator<<(std::ostream& os, const CPUMHATestCase& tc)
    {
        return os << "(batch_size: " << tc.batch_size << " sequence_count:" << tc.sequence_count
                  << " num_heads:" << tc.num_heads << " sequence_length:" << tc.sequence_length
                  << " problem_dimension:" << tc.problem_dimension
                  << " drop_out_rate:" << tc.drop_out_rate << " )";
    }
};
std::vector<CPUMHATestCase> CPUMHAConfigs()
{
    return {
        {// batch_size, sequence_count, num_heads, sequence_length, problem_dimension, drop_out_rate
         2,
         1,
         2,
         5,
         4,
         0.0}};
}

template <typename T = float>
struct CPUMHATest : public ::testing::TestWithParam<CPUMHATestCase>
{
protected:
    void SetUp() override
    {
        cpu_mha_test_case = GetParam();
        d_k               = cpu_mha_test_case.problem_dimension / cpu_mha_test_case.num_heads;
        // Initialize the tensors
        init();
        // mult-head attention begin

        // create Q, K and V
        Dot_4D_3D(word_position, q_weights, q_val);
        Dot_4D_3D(word_position, k_weights, k_val);
        Dot_4D_3D(word_position, v_weights, v_val);

        // Reshape K
        Transpose_5D(k_val, k_transpose);

        // Attention
        Dot_5D_5D(q_val, k_transpose, q_dot_k_transpose);

        // Attention Scale
        double sqrt_dk = 1.0 / std::sqrt(d_k);
        Scale(q_dot_k_transpose, sqrt_dk);
        // print5(q_dot_k_transpose, "q_dot_k_transpose");

        // AddMask5D_2D(q_dot_k_transpose, mask);

        // *** seperate softmax operation
        // double fp8_descale = 1.0 / 2.0;
        // // //   descale Q
        // Scale(q_val, fp8_descale);
        // // //   descale K
        // Scale(k_val, fp8_descale);

        // soft-max
        {
            // Row Reduction Max => M
            RowReductionMax(q_dot_k_transpose, rrm);

            // rrm substraction
            Sub(q_dot_k_transpose, rrm);

            // pointwise exponentiation
            Exponent(q_dot_k_transpose);

            Zinv(q_dot_k_transpose, zinv_tensors);

            // Zinv reciprocal
            ZinvMultiply(q_dot_k_transpose, zinv_tensors);
        }

        // drop out
        // DropOut(q_dot_k_transpose, cpu_mha_test_case.drop_out_rate);

        // drop out scalse
        double drop_out_scale = 1.0 / (1.0 - cpu_mha_test_case.drop_out_rate);
        Scale(q_dot_k_transpose, drop_out_scale);

        // print5(q_dot_k_transpose, "softrmaxxx");

        // descale Q
        // double scale_s = 1.0;
        // Scale(q_dot_k_transpose, scale_s);

        // O = (Q.dot(Kt)).dot(V)
        Dot_5D_5D(q_dot_k_transpose, v_val, atten_heads);
        // print5(atten_heads, "atten_heads ");
    }

    void TearDown() override
    {
        // verify
    }

private:
    void init()
    {
        mask = tensor<T>{
            std::vector<int>{cpu_mha_test_case.sequence_length, cpu_mha_test_case.sequence_length}};
        SetupMask(mask);

        q_val = tensor<T>{cpu_mha_test_case.batch_size,
                          cpu_mha_test_case.sequence_count,
                          cpu_mha_test_case.num_heads,
                          cpu_mha_test_case.sequence_length,
                          d_k};
        k_val = q_val;
        v_val = q_val;

        q_weights = tensor<T>(std::vector<int>{
            cpu_mha_test_case.num_heads, cpu_mha_test_case.problem_dimension, d_k});
        k_weights = q_weights;
        v_weights = q_weights;

        atten_heads       = tensor<T>{cpu_mha_test_case.batch_size,
                                cpu_mha_test_case.sequence_count,
                                cpu_mha_test_case.num_heads,
                                cpu_mha_test_case.sequence_length,
                                d_k};
        final_atten_heads = atten_heads;

        k_transpose = tensor<T>{cpu_mha_test_case.batch_size,
                                cpu_mha_test_case.sequence_count,
                                cpu_mha_test_case.num_heads,
                                d_k,
                                cpu_mha_test_case.sequence_length};

        q_dot_k_transpose = tensor<T>{cpu_mha_test_case.batch_size,
                                      cpu_mha_test_case.sequence_count,
                                      cpu_mha_test_case.num_heads,
                                      cpu_mha_test_case.sequence_length,
                                      cpu_mha_test_case.sequence_length};
        // reduce row max
        rrm = tensor<T>{cpu_mha_test_case.batch_size,
                        cpu_mha_test_case.sequence_count,
                        cpu_mha_test_case.num_heads,
                        cpu_mha_test_case.sequence_length,
                        1};
        //
        zinv_tensors = tensor<T>{cpu_mha_test_case.batch_size,
                                 cpu_mha_test_case.sequence_count,
                                 cpu_mha_test_case.num_heads,
                                 cpu_mha_test_case.sequence_length,
                                 1};

        word_position = tensor<T>{cpu_mha_test_case.batch_size,
                                  cpu_mha_test_case.sequence_count,
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
    tensor<T> final_atten_heads;

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
