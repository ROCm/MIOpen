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
#include <nlohmann/json.hpp>

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/layernorm.hpp>

#include "tensor_holder.hpp"
#include "tensor_util.hpp"
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

template <typename T>
void Dot_3D_3D_T(const tensor<T>& A_mat, const tensor<T>& B_mat, tensor<T>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[2];
    assert(k_val == B_mat.desc.GetLengths()[2]);
    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += A_mat(b_id, sl_id, k_id) * B_mat(h_id, dk_id, k_id);
        }
        C_mat(b_id, h_id, sl_id, dk_id) = T(sum);
    });
}

template <typename T1, typename T2>
void Dot_4D_4D_T(const tensor<T1>& A_mat, const tensor<T1>& B_mat, tensor<T2>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[3];
    assert(k_val == B_mat.desc.GetLengths()[3]); // since transpose
    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
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

template <typename T1, typename T2>
void Dot_3D_2D_T(const tensor<T1>& A_mat, const tensor<T1>& B_mat, tensor<T2>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[2];
    assert(k_val == B_mat.desc.GetLengths()[1]);
    C_mat.par_for_each([&](size_t b_id, size_t s_id, size_t pd_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += double(A_mat(b_id, s_id, k_id)) * double(B_mat(pd_id, k_id));
        }

        C_mat(b_id, s_id, pd_id) = T2(sum);
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

template <class T>
void Concat(const tensor<T>& A_mat, tensor<T>& B_mat)
{
    const auto& dims = A_mat.desc.GetLengths();
    size_t d_k       = dims[3];

    A_mat.par_for_each([&](size_t b_id, size_t h_id, size_t s_id, size_t dk_id) {
        B_mat(b_id, s_id, h_id * d_k + dk_id) = A_mat(b_id, h_id, s_id, dk_id);
    });
}

template <typename T>
void SoftMax(tensor<T>& q_dot_k_transpose, tensor<T>& rrm, tensor<T>& zinv_tensors)
{
    RowReductionMax(q_dot_k_transpose, rrm);
    // rrm substraction
    Sub(q_dot_k_transpose, rrm);
    // pointwise exponentiation
    Exponent(q_dot_k_transpose);
    Zinv(q_dot_k_transpose, zinv_tensors);
    // Zinv reciprocal
    ZinvMultiply(q_dot_k_transpose, zinv_tensors);
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
    // get fp8 version of Q and V
    Scale4DFp32ToFP8(q_val, q_val_fp8, q_scale);
    Scale4DFp32ToFP8(k_val, k_val_fp8, k_scale);

    // do fp8 BMM and store the fp8 result in fp32 tensor
    Dot_4D_4D_T(q_val_fp8, k_val_fp8, q_dot_k_transpose_fp32);

    // Scale to bring the fp8 values to fp32
    ScaleToFP32(q_dot_k_transpose_fp32, q_scale);
    ScaleToFP32(q_dot_k_transpose_fp32, k_scale);

    SoftMax(q_dot_k_transpose_fp32, rrm, zinv_tensors);

    // drop out
    // DropOut(q_dot_k_transpose, cpu_mha_test_case.drop_out_rate);

    tensor<float8> q_dot_k_transpose_fp8(q_dot_k_transpose.desc.GetLengths());
    tensor<float8> v_val_fp8(v_val.desc.GetLengths());

    // Get scaling of q_dot_k_transpose_fp32(S aka Softmax(Q.dot(K_transpose)))
    double AMax_S  = FindMax4D(q_dot_k_transpose_fp32);
    double s_scale = GetF8Scaling(AMax_S);
    // Get scaling of V
    double v_scale = GetF8Scaling(FindMax4D(v_val));

    // get fp8 version of S (Softmax(Q.dot(K_transpose))) and V
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

    SoftMax(q_dot_k_transpose, rrm, zinv_tensors);

    // // drop out
    // // DropOut(q_dot_k_transpose, cpu_mha_test_case.drop_out_rate);

    // // // drop out scalse
    // // double drop_out_scale = 1.0 / (1.0 - cpu_mha_test_case.drop_out_rate);
    // // Scale(q_dot_k_transpose, drop_out_scale);

    // // O = (Q.dot(Kt)).dot(V)
    Dot_4D_4D(q_dot_k_transpose, v_val, atten_heads);
}

template <typename T>
void ExtractGoldenDataFromJson(const std::string& test_file_name, tensor<T>& attention_golden)
{

    std::ifstream test_file(test_file_name);
    if(!test_file.is_open())
    {
        std::cerr << "Cannot find " << test_file_name << std::endl;
        exit(1);
    }
    nlohmann::json jsonTensor;
    test_file >> jsonTensor;
    test_file.close();
    std::vector<std::vector<float>> tensorData =
        jsonTensor["tensor"].get<std::vector<std::vector<float>>>();
    // Check if the "tensor" key exists and is an array
    if(!jsonTensor.contains("tensor") || !jsonTensor["tensor"].is_array())
    {
        std::cerr << "'tensor' key not found or is not an array" << std::endl;
        exit(1);
    }
    // Extract the 2D array and flatten it
    std::vector<float> flatTensor;
    for(const auto& row : jsonTensor["tensor"])
    {
        if(!row.is_array())
        {
            std::cerr << "Expected a row to be an array, but found a different type" << std::endl;
            exit(1);
        }
        for(const auto& val : row)
        {
            // Ensure each value is a number before adding it to the flatTensor
            if(!val.is_number())
            {
                std::cerr << "Expected a value to be a number, but found a different type"
                          << std::endl;
                exit(1);
            }
            flatTensor.push_back(val.get<float>());
        }
    }
    attention_golden.data = flatTensor;
}

} // namespace cpu
} // namespace test
