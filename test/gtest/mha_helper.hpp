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

namespace test {
namespace cpu {

using float8 = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;

struct CPUMHATestCase
{
    // represents total number of seq len present in batch
    size_t batch_size;
    size_t sequence_length;
    size_t num_heads;
    size_t problem_dimension;
    float drop_out_rate;

    friend std::ostream& operator<<(std::ostream& os, const CPUMHATestCase& tc)
    {
        return os << "(batch_size: " << tc.batch_size << " num_heads:" << tc.num_heads
                  << " sequence_length:" << tc.sequence_length
                  << " problem_dimension:" << tc.problem_dimension
                  << " drop_out_rate:" << tc.drop_out_rate << " )";
    }
};

double GetF8Scaling(double max_val)
{
    const double fp8_E4M3_max = 240.0f;

    return fp8_E4M3_max / max_val;
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
                    std::cout << std::fixed << std::setprecision(10) << tensor_val(i, j, k, l)
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

// Todo : make this abs
template <typename T>
T FindMax4D(const tensor<T>& max_of_tensor)
{
    std::mutex mtx;
    T maxVal = max_of_tensor(0, 0, 0, 0); // Start with the first element as the maximum
    max_of_tensor.par_for_each([&](size_t b_id, size_t n_id, size_t s_id, size_t dk_id) {
        std::lock_guard<std::mutex> lock(mtx);
        T tmp_val = max_of_tensor(b_id, n_id, s_id, dk_id);
        if(tmp_val > maxVal)
        {
            maxVal = tmp_val;
        }
    });
    return maxVal;
}

// C_mat = A_mat.dot(B_mat)
// A_mat : 3D
// B_mat : 3D
// C_mat : 4D
template <typename T>
void Dot_3D_3D(const tensor<T>& A_mat, const tensor<T>& B_mat, tensor<T>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[2];
    assert(k_val == B_mat.desc.GetLengths()[1]);
    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += A_mat(b_id, sl_id, k_id) * B_mat(h_id, k_id, dk_id);
        }
        C_mat(b_id, h_id, sl_id, dk_id) = T(sum);
    });
}

// C_mat = A_mat.dot(transpose(B_mat))
// A_mat : 4D
// B_mat : 4D
// C_mat : 4D
template <typename T1, typename T2>
void Dot_4D_4D_T(const tensor<T1>& A_mat, const tensor<T1>& B_mat, tensor<T2>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[3];
    assert(k_val == B_mat.desc.GetLengths()[3]); // since transpose
    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        T2 sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += T2(A_mat(b_id, h_id, sl_id, k_id)) * T2(B_mat(b_id, h_id, dk_id, k_id));
        }

        C_mat(b_id, h_id, sl_id, dk_id) = T2(sum);
    });
}

template <typename T1, typename T2>
void Dot_4D_4D(const tensor<T1>& A_mat, const tensor<T1>& B_mat, tensor<T2>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[3];
    assert(k_val == B_mat.desc.GetLengths()[2]);
    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += double(A_mat(b_id, h_id, sl_id, k_id)) * double(B_mat(b_id, h_id, k_id, dk_id));
        }

        C_mat(b_id, h_id, sl_id, dk_id) = T2(sum);
    });
}

template <typename T>
void AddMask4D_2D(tensor<T>& mat_A_val, const tensor<T>& mat_mask)
{
    mat_A_val.par_for_each([&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
        mat_A_val(b_id, h_id, sl_i_id, sl_j_id) =
            mat_A_val(b_id, h_id, sl_i_id, sl_j_id) + mat_mask(sl_i_id, sl_j_id);
    });
}

// Computes the sum of the row in A_mat
// A_mat : 4D
// rrm_tensor : 4D
template <class T>
void RowReductionMax(const tensor<T>& A_mat, tensor<T>& rrm_tensor)
{
    size_t sl_dim = A_mat.desc.GetLengths()[3];
    rrm_tensor.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t sl0_id) {
        T max(A_mat(b_id, h_id, sl_id, sl0_id));
        for(size_t id = 0; id < sl_dim; ++id)
        {
            if(A_mat(b_id, h_id, sl_id, id) > max)
            {
                max = A_mat(b_id, h_id, sl_id, id);
            }
        }
        rrm_tensor(b_id, h_id, sl_id, sl0_id) = max;
    });
}

template <class T>
void Zinv(const tensor<T>& A_mat, tensor<T>& zinv_tensor)
{
    size_t sl_dim = A_mat.desc.GetLengths()[3];
    T one(1);
    zinv_tensor.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t sl0_id) {
        double sum(0);
        for(size_t id = 0; id < sl_dim; ++id)
        {
            sum += A_mat(b_id, h_id, sl_id, id);
        }
        zinv_tensor(b_id, h_id, sl_id, sl0_id) = T(one / sum);
    });
}

template <typename T>
void Scale4DFp32ToFP8(const tensor<T>& mat_val, tensor<float8>& mat_val_fp8, double scale_factor)
{
    mat_val_fp8.par_for_each([&](size_t b_id, size_t sl_i_id, size_t sl_j_id, size_t sl_k_id) {
        mat_val_fp8(b_id, sl_i_id, sl_j_id, sl_k_id) =
            float8(mat_val(b_id, sl_i_id, sl_j_id, sl_k_id) * scale_factor);
    });
}

template <typename T>
void ScaleToFP32(tensor<T>& mat_val, double scale_factor)
{
    mat_val.par_for_each([&](size_t b_id, size_t sl_i_id, size_t sl_j_id, size_t sl_k_id) {
        mat_val(b_id, sl_i_id, sl_j_id, sl_k_id) =
            T(T(mat_val(b_id, sl_i_id, sl_j_id, sl_k_id)) / scale_factor);
    });
}

template <typename T>
void Sub(tensor<T>& mat_A_val, const tensor<T>& mat_sub)
{
    mat_A_val.par_for_each([&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
        mat_A_val(b_id, h_id, sl_i_id, sl_j_id) =
            mat_A_val(b_id, h_id, sl_i_id, sl_j_id) - mat_sub(b_id, h_id, sl_i_id, 0);
    });
}

template <class T>
void Exponent(tensor<T>& tensor_val)
{
    tensor_val.par_for_each([&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
        tensor_val(b_id, h_id, sl_i_id, sl_j_id) =
            T(std::exp(tensor_val(b_id, h_id, sl_i_id, sl_j_id)));
    });
}

template <class T>
void ZinvMultiply(tensor<T>& q_dot_k_transpose, const tensor<T>& zinv_tensors)
{
    q_dot_k_transpose.par_for_each([&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
        q_dot_k_transpose(b_id, h_id, sl_i_id, sl_j_id) =
            q_dot_k_transpose(b_id, h_id, sl_i_id, sl_j_id) * zinv_tensors(b_id, h_id, sl_i_id, 0);
    });
}

// template <typename T>
// void SetupMask(tensor<T>& mask)
// {
//     mask.par_for_each([&](size_t s_id, size_t p_id) {
//         // make anything above diagonal inf
//         if(p_id > s_id)
//         {
//             mask(s_id, p_id) = -std::numeric_limits<T>::infinity();
//         }
//     });
// }

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

template <typename T>
void MultiHeadAttentionfp8(tensor<T>& q_val,
                           tensor<T>& k_val,
                           tensor<T>& v_val,
                           tensor<T>& q_dot_k_transpose,
                           tensor<T>& rrm,
                           tensor<T>& zinv_tensors,
                           tensor<float8>& atten_heads_fp8)
{
    tensor<float8> q_val_fp8(q_val.desc.GetLengths());
    tensor<float8> k_val_fp8(k_val.desc.GetLengths());
    tensor<T> q_dot_k_transpose_fp32(q_dot_k_transpose.desc.GetLengths());

    double q_scale = GetF8Scaling(FindMax4D(q_val));
    double k_scale = GetF8Scaling(FindMax4D(k_val));
    Scale4DFp32ToFP8(q_val, q_val_fp8, q_scale);
    Scale4DFp32ToFP8(k_val, k_val_fp8, k_scale);

    Dot_4D_4D_T(q_val_fp8, k_val_fp8, q_dot_k_transpose_fp32);

    ScaleToFP32(q_dot_k_transpose_fp32, q_scale);
    ScaleToFP32(q_dot_k_transpose_fp32, k_scale);

    // soft-max
    {
        // Row Reduction Max => M
        RowReductionMax(q_dot_k_transpose_fp32, rrm);
        // rrm substraction
        Sub(q_dot_k_transpose, rrm);

        // pointwise exponentiation
        Exponent(q_dot_k_transpose_fp32);
        Zinv(q_dot_k_transpose_fp32, zinv_tensors);
        // Zinv reciprocal
        ZinvMultiply(q_dot_k_transpose_fp32, zinv_tensors);
    }

    // drop out
    // DropOut(q_dot_k_transpose, cpu_mha_test_case.drop_out_rate);

    tensor<float8> q_dot_k_transpose_fp8(q_dot_k_transpose.desc.GetLengths());
    tensor<float8> v_val_fp8(v_val.desc.GetLengths());
    double AMax_S  = FindMax4D(q_dot_k_transpose_fp32);
    double s_scale = GetF8Scaling(AMax_S);
    double v_scale = GetF8Scaling(FindMax4D(v_val));

    Scale4DFp32ToFP8(q_dot_k_transpose_fp32, q_dot_k_transpose_fp8, s_scale);
    Scale4DFp32ToFP8(v_val, v_val_fp8, v_scale);

    tensor<T> atten_heads_fp32(atten_heads_fp8.desc.GetLengths());

    Dot_4D_4D(q_dot_k_transpose_fp8, v_val_fp8, atten_heads_fp32);

    ScaleToFP32(atten_heads_fp32, s_scale);
    ScaleToFP32(atten_heads_fp32, v_scale);
    double scale_O = GetF8Scaling(FindMax4D(atten_heads_fp32));

    Scale4DFp32ToFP8(atten_heads_fp32, atten_heads_fp8, scale_O);
}

template <typename T>
void MultiHeadAttentionf32(tensor<T>& q_val,
                           tensor<T>& k_val,
                           tensor<T>& v_val,
                           tensor<T>& q_dot_k_transpose,
                           tensor<T>& rrm,
                           tensor<T>& zinv_tensors,
                           tensor<T>& atten_heads)
{

    Dot_4D_4D_T(q_val, k_val, q_dot_k_transpose);

    // soft-max
    {
        // Row Reduction Max => M
        RowReductionMax(q_dot_k_transpose, rrm);

        // rrm substraction
        // Sub(q_dot_k_transpose, rrm);

        // pointwise exponentiation
        Exponent(q_dot_k_transpose);

        Zinv(q_dot_k_transpose, zinv_tensors);

        // Zinv reciprocal
        ZinvMultiply(q_dot_k_transpose, zinv_tensors);
    }

    // // drop out
    // // DropOut(q_dot_k_transpose, cpu_mha_test_case.drop_out_rate);

    // // // drop out scalse
    // // double drop_out_scale = 1.0 / (1.0 - cpu_mha_test_case.drop_out_rate);
    // // Scale(q_dot_k_transpose, drop_out_scale);

    // // O = (Q.dot(Kt)).dot(V)
    Dot_4D_4D(q_dot_k_transpose, v_val, atten_heads);
}

} // namespace cpu
} // namespace test
