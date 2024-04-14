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

#include <hip_float8.hpp>
#include <nlohmann/json.hpp>

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

float GetF8Scaling(float max_val)
{
    constexpr float fp8_E4M3_max = 240.0f;

    return fp8_E4M3_max / max_val;
}

template <typename T>
T FindMax4D(const tensor<T>& max_of_tensor)
{
    T maxVal = max_of_tensor(0, 0, 0, 0); // Start with the first element as the maximum
    max_of_tensor.for_each([&](size_t b_id, size_t n_id, size_t s_id, size_t dk_id) {
        maxVal = std::max(maxVal, max_of_tensor(b_id, n_id, s_id, dk_id));
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
void Dot_4D_T_4D(const tensor<T1>& A_mat, const tensor<T1>& B_mat, tensor<T2>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[3];
    assert(k_val == B_mat.desc.GetLengths()[2]); // since transpose

    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += T2(A_mat(b_id, h_id, k_id, sl_id)) * T2(B_mat(b_id, h_id, k_id, dk_id));
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
        mat_A_val(b_id, h_id, sl_i_id, sl_j_id) += mat_mask(sl_i_id, sl_j_id);
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
            max = std::max(max, A_mat(b_id, h_id, sl_id, id));
        }
        rrm_tensor(b_id, h_id, sl_id, sl0_id) = max;
    });
}

template <typename T1, typename T2, typename T3>
void ScaleMult(const tensor<T1>& tensor_val,
               const T2& scale_factor,
               tensor<T3>& tensor_scale_factor)
{
    tensor_scale_factor.par_for_each([&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
        tensor_scale_factor(b_id, h_id, sl_i_id, sl_j_id) =
            T3(tensor_val(b_id, h_id, sl_i_id, sl_j_id) * scale_factor);
    });
}

template <typename T1, typename T2, typename T3>
void ScaleMult3d(const tensor<T1>& tensor_val,
                 const T2& scale_factor,
                 tensor<T3>& tensor_scale_factor)
{
    tensor_scale_factor.par_for_each([&](size_t b_id, size_t sl_i_id, size_t sl_j_id) {
        tensor_scale_factor(b_id, sl_i_id, sl_j_id) =
            T3(tensor_val(b_id, sl_i_id, sl_j_id) * scale_factor);
    });
}

template <class T>
void PointWiseExp(const tensor<T>& tensor_val, tensor<T>& tensor_exp_val)
{
    tensor_exp_val.par_for_each([&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
        tensor_exp_val(b_id, h_id, sl_i_id, sl_j_id) =
            T(std::exp(tensor_val(b_id, h_id, sl_i_id, sl_j_id)));
    });
}

template <class T>
void PointWiseMultiply(const tensor<T>& tensor_a, const tensor<T>& tensor_b, tensor<T>& tensor_c)
{
    tensor_c.par_for_each([&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
        tensor_c(b_id, h_id, sl_i_id, sl_j_id) =
            tensor_a(b_id, h_id, sl_i_id, sl_j_id) * tensor_b(b_id, h_id, sl_i_id, sl_j_id);
    });
}

template <typename T>
void BroadCastSub(const tensor<T>& tensor_val1,
                  const tensor<T>& tesnor_val2,
                  tensor<T>& tensor_val1_sub_val2)
{
    tensor_val1_sub_val2.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            tensor_val1_sub_val2(b_id, h_id, sl_i_id, sl_j_id) =
                tensor_val1(b_id, h_id, sl_i_id, sl_j_id) - tesnor_val2(b_id, h_id, sl_i_id, 0);
        });
}

template <typename T>
void BroadCastAdd(const tensor<T>& tensor_val1,
                  const tensor<T>& tesnor_val2,
                  tensor<T>& tensor_val1_sub_val2)
{
    tensor_val1_sub_val2.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            tensor_val1_sub_val2(b_id, h_id, sl_i_id, sl_j_id) =
                tensor_val1(b_id, h_id, sl_i_id, sl_j_id) + tesnor_val2(b_id, h_id, sl_i_id, 0);
        });
}

template <class T>
void BroadCastMul(const tensor<T>& tensor_val, const tensor<T>& z_sum, tensor<T>& tensor_div_z_sum)
{
    tensor_div_z_sum.par_for_each([&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
        tensor_div_z_sum(b_id, h_id, sl_i_id, sl_j_id) =
            tensor_val(b_id, h_id, sl_i_id, sl_j_id) * z_sum(b_id, h_id, sl_i_id, 0);
    });
}

template <class T>
void RowReductionReciprocalSum(const tensor<T>& A_mat, tensor<T>& rrsum_tensor)
{
    size_t sl_dim = A_mat.desc.GetLengths()[3];
    rrsum_tensor.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t sl0_id) {
        double sum(0);
        for(size_t id = 0; id < sl_dim; ++id)
        {
            sum += A_mat(b_id, h_id, sl_id, id);
        }
        rrsum_tensor(b_id, h_id, sl_id, sl0_id) = static_cast<T>(1) / sum;
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

template <class T, class U>
void Concat(const tensor<T>& A_mat, tensor<U>& B_mat)
{
    const auto& dims = A_mat.desc.GetLengths();
    size_t d_k       = dims[3];

    A_mat.par_for_each([&](size_t b_id, size_t h_id, size_t s_id, size_t dk_id) {
        B_mat(b_id, s_id, h_id * d_k + dk_id) = A_mat(b_id, h_id, s_id, dk_id);
    });
}

/* attn_max  : max_of_each_row_of(q_dot_k_transpose)
 *              Its a row reduction operation. eg: (3x3) => (3x1)
 * z_sum      : sum(exp((q_dot_k_transpose - attn_max))) eg: (3x3) => (3x1)
 *              Its a row reduction operation. eg: (3x3) => (3x1)
 */
template <typename T>
void SoftMax(const tensor<T>& q_dot_k_transpose,
             tensor<T>& softmax,
             tensor<T>& attn_max,
             tensor<T>& z_sum)
{
    // compute max across each row of matrix. This value is
    // used for numerical stability for softmax computation.
    RowReductionMax(q_dot_k_transpose, attn_max);

    // substract the computed max
    auto q_dot_k_transpose_sub_attn_max = q_dot_k_transpose;
    BroadCastSub(q_dot_k_transpose, attn_max, q_dot_k_transpose_sub_attn_max);

    // Compute the exponential of each element
    // exp(row_max)
    auto exp_q_dot_k_transpose_sub_attn_max = q_dot_k_transpose_sub_attn_max;
    PointWiseExp(q_dot_k_transpose_sub_attn_max, exp_q_dot_k_transpose_sub_attn_max);

    // z_sum aka attn_norm = 1 / sum(exp((q_dot_k_transpose - attn_max)))
    RowReductionReciprocalSum(exp_q_dot_k_transpose_sub_attn_max, z_sum);

    // softmax = exp((q_dot_k_transpose - attn_max)) * z_sum
    BroadCastMul(exp_q_dot_k_transpose_sub_attn_max, z_sum, softmax);
}

template <typename T>
void MultiHeadAttentionfp8(const tensor<T>& q_val,
                           const tensor<T>& k_val,
                           const tensor<T>& v_val,
                           tensor<T>& softmax,
                           tensor<T>& attn_max,
                           tensor<T>& Z_sum,
                           float& q_scale,
                           float& k_scale,
                           float& aMax_S,
                           float& s_scale,
                           float& v_scale,
                           float& scale_O,
                           tensor<float8>& multi_head_attention_fp8)
{
    tensor<float8> q_val_fp8(q_val.desc.GetLengths());
    tensor<float8> k_val_fp8(k_val.desc.GetLengths());
    tensor<T> q_dot_k_fp8_stored_in_fp32_tensor(softmax.desc.GetLengths());

    q_scale = GetF8Scaling(FindMax4D(q_val)); // (max fp8 can represent)/(max val in q_val)
    k_scale = GetF8Scaling(FindMax4D(k_val)); // (max fp8 can represent)/(max val in k_val)
    // scale fp32 vals to get fp8 version of Q and V
    ScaleMult(q_val, q_scale, q_val_fp8);
    ScaleMult(k_val, k_scale, k_val_fp8);

    // The FP8 Matrix Multiplicatin happens here.
    // The results are tored in fp32 tensor.
    // 1) fp8 matrix multiplication
    Dot_4D_4D_T(q_val_fp8, k_val_fp8, q_dot_k_fp8_stored_in_fp32_tensor);

    // bring it back to fp32 so that we can do the softmax
    ScaleMult(q_dot_k_fp8_stored_in_fp32_tensor, 1.0 / q_scale, q_dot_k_fp8_stored_in_fp32_tensor);
    ScaleMult(q_dot_k_fp8_stored_in_fp32_tensor, 1.0 / k_scale, q_dot_k_fp8_stored_in_fp32_tensor);

    SoftMax(q_dot_k_fp8_stored_in_fp32_tensor, softmax, attn_max, Z_sum);

    // drop out
    // DropOut(q_dot_k_transpose, cpu_mha_test_case.drop_out_rate);

    // Get scaling for softmax
    aMax_S  = FindMax4D(softmax);
    s_scale = GetF8Scaling(aMax_S);
    // Get scaling of V
    v_scale = GetF8Scaling(FindMax4D(v_val));

    tensor<float8> softmax_fp8(softmax.desc.GetLengths());
    tensor<float8> v_val_fp8(v_val.desc.GetLengths());

    // get fp8 version of Softmax(Q.dot(K_transpose)) and V
    ScaleMult(softmax, s_scale, softmax_fp8);
    ScaleMult(v_val, v_scale, v_val_fp8);

    tensor<T> atten_heads_fp32(multi_head_attention_fp8.desc.GetLengths());

    // 2) fp8 matrix multiplication
    Dot_4D_4D(softmax_fp8, v_val_fp8, atten_heads_fp32);

    // bring it back to fp32
    ScaleMult(atten_heads_fp32, 1.0 / s_scale, atten_heads_fp32);
    ScaleMult(atten_heads_fp32, 1.0 / v_scale, atten_heads_fp32);
    scale_O = GetF8Scaling(FindMax4D(atten_heads_fp32));

    // scale to fp8 version
    ScaleMult(atten_heads_fp32, scale_O, multi_head_attention_fp8);
}

template <typename T>
void MultiHeadAttentionf32(const tensor<T>& q_val,
                           const tensor<T>& k_val,
                           const tensor<T>& v_val,
                           tensor<T>& q_dot_k_transpose,
                           tensor<T>& softmax,
                           tensor<T>& attn_max,
                           tensor<T>& Z_sum,
                           tensor<T>& multi_head_attention)
{

    Dot_4D_4D_T(q_val, k_val, q_dot_k_transpose);

    SoftMax(q_dot_k_transpose, softmax, attn_max, Z_sum /*attn_norm*/);

    // // drop out
    // // DropOut(softmax, cpu_mha_test_case.drop_out_rate);

    // // // drop out scalse
    // // double drop_out_scale = 1.0 / (1.0 - cpu_mha_test_case.drop_out_rate);
    // // Scale(softmax, drop_out_scale);

    // O = (Q.dot(Kt)).dot(V)
    Dot_4D_4D(softmax, v_val, multi_head_attention);
}

template <typename T>
tensor<float> ExtractGoldenDataFromJson(std::string_view json_attention_data,
                                        const tensor<T>& tensor_val)
{
    auto jsonTensor = nlohmann::json::parse(json_attention_data);
    // Check if the "tensor" key exists and is an array
    EXPECT_TRUE(jsonTensor.contains("tensor") && jsonTensor["tensor"].is_array())
        << "Malformed ref data: 'tensor' key not found or is not an array";

    // Extract the 2D array and flatten it
    tensor<float> res(tensor_val.desc);
    res.data.clear(); // tensor constructed with .resize(), but we need push_back
    for(const auto& row : jsonTensor["tensor"])
    {
        EXPECT_TRUE(row.is_array())
            << "Malformed ref data: expected a row to be an array, but found a different type";
        for(const auto& val : row)
        {
            EXPECT_TRUE(val.is_number()) << "Malformed ref data: expected a value to be a "
                                            "number, but found a different type";
            res.data.emplace_back(val.get<float>());
        }
    }

    EXPECT_EQ(res.data.size(), tensor_val.data.size())
        << "Malformed ref data: reference tensor has different size";

    return res;
}

} // namespace cpu
} // namespace test
