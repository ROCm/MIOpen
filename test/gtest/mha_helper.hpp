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

// disable __device__ qualifiers
#ifdef FQUALIFIERS
#error rocrand FQUALIFIERS defined externally, probably one of rocrand device header included prior to this
#endif
#define FQUALIFIERS inline
#include <miopen_rocrand.hpp>

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

inline constexpr float GetF8Scaling(float max_val)
{
    constexpr float fp8_E4M3_max = 240.0f;

    return fp8_E4M3_max / max_val;
}

template <typename T>
T AbsoluteMax(const tensor<T>& max_of_tensor)
{
    T maxVal = std::abs(max_of_tensor[0]); // Start with the first element as the maximum
    max_of_tensor.for_each(
        [&](auto... id) { maxVal = std::max(maxVal, std::abs(max_of_tensor(id...))); });
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

template <typename T1, typename T2, typename T3 = T1>
void Dot_4D_4D_T(const tensor<T1>& A_mat, const tensor<T3>& B_mat, tensor<T2>& C_mat)
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

template <typename T1, typename T2, typename T3 = T1>
void Dot_4D_T_4D(const tensor<T1>& A_mat, const tensor<T3>& B_mat, tensor<T2>& C_mat)
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

template <typename T1, typename T2, typename T3>
void Dot_4D_4D(const tensor<T1>& A_mat, const tensor<T2>& B_mat, tensor<T3>& C_mat)
{
    size_t k_val = A_mat.desc.GetLengths()[3];
    assert(k_val == B_mat.desc.GetLengths()[2]);
    C_mat.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t dk_id) {
        double sum(0);
        for(size_t k_id = 0; k_id < k_val; ++k_id)
        {
            sum += double(A_mat(b_id, h_id, sl_id, k_id)) * double(B_mat(b_id, h_id, k_id, dk_id));
        }

        C_mat(b_id, h_id, sl_id, dk_id) = T3(sum);
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
    tensor_scale_factor.par_for_each(
        [&](auto... id) { tensor_scale_factor(id...) = T3(tensor_val(id...) * scale_factor); });
}

template <class T>
void PointWiseExp(const tensor<T>& tensor_val, tensor<T>& tensor_exp_val)
{
    tensor_exp_val.par_for_each(
        [&](auto... id) { tensor_exp_val(id...) = T(std::exp(tensor_val(id...))); });
}

template <class T>
void PointWiseMultiply(const tensor<T>& tensor_a, const tensor<T>& tensor_b, tensor<T>& tensor_c)
{
    tensor_c.par_for_each([&](auto... id) { tensor_c(id...) = tensor_a(id...) * tensor_b(id...); });
}

template <class T>
void PointWiseAdd(const tensor<T>& tensor_a, const tensor<T>& tensor_b, tensor<T>& tensor_c)
{
    tensor_c.par_for_each([&](auto... id) { tensor_c(id...) = tensor_a(id...) + tensor_b(id...); });
}

template <typename T>
void BroadCastSub(const tensor<T>& tensor_val1,
                  const tensor<T>& tensor_val2,
                  tensor<T>& tensor_val1_sub_val2)
{
    tensor_val1_sub_val2.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            tensor_val1_sub_val2(b_id, h_id, sl_i_id, sl_j_id) =
                tensor_val1(b_id, h_id, sl_i_id, sl_j_id) - tensor_val2(b_id, h_id, sl_i_id, 0);
        });
}

template <typename T>
void BroadCastAdd(const tensor<T>& tensor_val1,
                  const tensor<T>& tensor_val2,
                  tensor<T>& tensor_val1_sub_val2)
{
    tensor_val1_sub_val2.par_for_each(
        [&](size_t b_id, size_t h_id, size_t sl_i_id, size_t sl_j_id) {
            tensor_val1_sub_val2(b_id, h_id, sl_i_id, sl_j_id) =
                tensor_val1(b_id, h_id, sl_i_id, sl_j_id) + tensor_val2(b_id, h_id, sl_i_id, 0);
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

template <class T>
void RowReductionSum(const tensor<T>& A_mat, tensor<T>& rrsum_tensor)
{
    size_t sl_dim = A_mat.desc.GetLengths()[3];
    rrsum_tensor.par_for_each([&](size_t b_id, size_t h_id, size_t sl_id, size_t sl0_id) {
        double sum(0);
        for(size_t id = 0; id < sl_dim; ++id)
        {
            sum += A_mat(b_id, h_id, sl_id, id);
        }
        rrsum_tensor(b_id, h_id, sl_id, sl0_id) = sum;
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
void DropOut(tensor<T>& tensor, float dropout_rate, uint64_t seed, uint64_t offset)
{
    if(dropout_rate > 0.0f)
    {
        // it assumes that 'blockIdx.x * blockDim.x + threadIdx.x' will cover whole tensor
        // and without idle threads
        std::vector<rocrand_state_xorwow> rng(tensor.data.size());
        for(size_t idx = 0; idx < tensor.data.size(); ++idx)
        {
            rocrand_init(prng::hash(seed + idx), 0, offset, &rng[idx]);
        }

        size_t idx = 0;
        tensor.for_each([&, scale = 1.0f / (1.0f - dropout_rate)](auto... id) {
            const bool drop = prng::xorwow_uniform(&rng[idx++]) < dropout_rate;
            tensor(id...)   = drop ? T(0) : tensor(id...) * scale;
        });
    }
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

template <typename T = float8>
void MultiHeadAttentionForwardfp8(const tensor<T>& q_val,
                                  const tensor<T>& k_val,
                                  const tensor<T>& v_val,
                                  tensor<float>& softmax,
                                  tensor<float>& attn_max,
                                  tensor<float>& Z_sum,
                                  float q_descale,
                                  float k_descale,
                                  float v_descale,
                                  float s_descale,
                                  float s_scale,
                                  float o_scale,
                                  float dropout_rate,
                                  uint64_t seed,
                                  uint64_t offset,
                                  float& aMax_S,
                                  float& aMax_O,
                                  tensor<T>& multi_head_attention_fp8,
                                  const tensor<float>* optional_bias =
                                      nullptr) // pointer to optional bias, nullptr if not provided
{
    auto inputLengths = q_val.desc.GetLengths();
    inputLengths[3]   = inputLengths[2]; // NHSD converting to NHSS
    tensor<float> q_dot_k_fp8_stored_in_fp32_tensor(inputLengths);

    // The FP8 Matrix Multiplicatin happens here.
    // The results are tored in fp32 tensor.
    // 1) fp8 matrix multiplication
    Dot_4D_4D_T(q_val, k_val, q_dot_k_fp8_stored_in_fp32_tensor);

    // bring it back to fp32 so that we can do the softmax
    ScaleMult(q_dot_k_fp8_stored_in_fp32_tensor,
              q_descale * k_descale,
              q_dot_k_fp8_stored_in_fp32_tensor);

    if(optional_bias != nullptr)
    {
        PointWiseAdd(
            q_dot_k_fp8_stored_in_fp32_tensor, *optional_bias, q_dot_k_fp8_stored_in_fp32_tensor);
    }

    SoftMax(q_dot_k_fp8_stored_in_fp32_tensor, softmax, attn_max, Z_sum);

    // Get scaling for softmax
    aMax_S = AbsoluteMax(softmax);

    // drop out
    DropOut(softmax, dropout_rate, seed, offset);

    tensor<T> softmax_fp8(softmax.desc.GetLengths());
    // get fp8 version of Softmax(Q.dot(K_transpose)) and V
    ScaleMult(softmax, s_scale, softmax_fp8);

    tensor<float> atten_heads_fp32(multi_head_attention_fp8.desc.GetLengths());
    // 2) fp8 matrix multiplication
    Dot_4D_4D(softmax_fp8, v_val, atten_heads_fp32);

    // bring it back to fp32
    ScaleMult(atten_heads_fp32, s_descale * v_descale, atten_heads_fp32);
    aMax_O = AbsoluteMax(atten_heads_fp32);

    // scale to fp8 version
    ScaleMult(atten_heads_fp32, o_scale, multi_head_attention_fp8);
}

template <typename T>
void MultiHeadAttentionForwardfp16(const tensor<T>& q_val,
                                   const tensor<T>& k_val,
                                   const tensor<T>& v_val,
                                   tensor<float>& softmax,
                                   tensor<float>& attn_max,
                                   tensor<float>& Z_sum,
                                   tensor<T>& multi_head_attention,
                                   const tensor<float>* optional_bias =
                                       nullptr) // pointer to optional bias, nullptr if not provided
{
    auto inputLengths = q_val.desc.GetLengths();
    inputLengths[3]   = inputLengths[2]; // NHSD converting to NHSS
    tensor<float> q_dot_k_transpose(inputLengths);

    Dot_4D_4D_T(q_val, k_val, q_dot_k_transpose);

    if(optional_bias != nullptr)
    {
        PointWiseAdd(q_dot_k_transpose, *optional_bias, q_dot_k_transpose);
    }

    SoftMax(q_dot_k_transpose, softmax, attn_max, Z_sum);

    Dot_4D_4D(softmax, v_val, multi_head_attention);
}

template <typename T>
void MultiHeadAttentionForwardf32(
    const tensor<T>& q_val,
    const tensor<T>& k_val,
    const tensor<T>& v_val,
    tensor<T>& q_dot_k_transpose,
    tensor<T>& softmax,
    tensor<T>& attn_max,
    tensor<T>& Z_sum,
    float& aMax_S,
    float& aMax_O,
    tensor<T>& multi_head_attention,
    const tensor<T>* optional_bias = nullptr) // pointer to optional bias, nullptr if not provided
{
    Dot_4D_4D_T(q_val, k_val, q_dot_k_transpose);

    if(optional_bias != nullptr)
    {
        PointWiseAdd(q_dot_k_transpose, *optional_bias, q_dot_k_transpose);
    }

    SoftMax(q_dot_k_transpose, softmax, attn_max, Z_sum);
    aMax_S = AbsoluteMax(softmax);

    // // drop out
    // // DropOut(softmax, cpu_mha_test_case.drop_out_rate);

    // // // drop out scalse
    // // double drop_out_scale = 1.0 / (1.0 - cpu_mha_test_case.drop_out_rate);
    // // Scale(softmax, drop_out_scale);

    // O = (Q.dot(Kt)).dot(V)
    Dot_4D_4D(softmax, v_val, multi_head_attention);
    aMax_O = AbsoluteMax(multi_head_attention);
}

template <typename T>
void MultiHeadAttentionBackwardDataf32(const tensor<T>& q_val,
                                       const tensor<T>& k_val,
                                       const tensor<T>& v_val,
                                       const tensor<T>& O_val, // attention (O)
                                       const tensor<T>& dO_val,
                                       const tensor<T>& softmax,
                                       const tensor<T>& attn_max,
                                       const tensor<T>& z_sum,
                                       tensor<T>& dQ_val,
                                       tensor<T>& dK_val,
                                       tensor<T>& dV_val)
{

    auto inputLengths = q_val.desc.GetLengths();
    inputLengths[3]   = inputLengths[2]; // NHSD converting to NHSS

    tensor<T> dO_dot_V_tranpsoe_val(inputLengths);
    tensor<T> bwd_intermediate(inputLengths);
    tensor<T> dO_pointwiseMul_O(dO_val.desc.GetLengths());
    inputLengths[3] = 1; // NHSD converting to NHS1 for row reduction tensor
    tensor<T> dO_pointwiseMul_O_rrsum(inputLengths);

    // softmax_T.dO

    // 1) fp8 matrix multiplication
    Dot_4D_T_4D(softmax, dO_val, dV_val);

    // dO x V
    Dot_4D_4D_T(dO_val, v_val, dO_dot_V_tranpsoe_val);

    // dO . O
    PointWiseMultiply(dO_val, O_val, dO_pointwiseMul_O);

    RowReductionSum(dO_pointwiseMul_O, dO_pointwiseMul_O_rrsum);
    BroadCastSub(dO_dot_V_tranpsoe_val, dO_pointwiseMul_O_rrsum, bwd_intermediate);
    PointWiseMultiply(bwd_intermediate, softmax, bwd_intermediate);

    // both dk and dQ
    // finally
    Dot_4D_4D(bwd_intermediate, k_val, dQ_val);
    Dot_4D_T_4D(bwd_intermediate, q_val, dK_val);
}

template <typename T = float8, typename U = T>
void MultiHeadAttentionBackwardDataf8(const tensor<T>& q_val,
                                      const tensor<T>& k_val,
                                      const tensor<T>& v_val,
                                      const tensor<T>& O_val, // attention (O)
                                      const tensor<U>& dO_val,
                                      const tensor<float>& softmax_fp32,
                                      float q_descale,
                                      float k_descale,
                                      float v_descale,
                                      float dQ_scale,
                                      float dK_scale,
                                      float dV_scale,
                                      float s_scale,
                                      float s_descale,
                                      float ds_scale,
                                      float ds_descale,
                                      float O_descale,
                                      float dO_descale,
                                      float& aMax_dS,
                                      float& aMax_dQ,
                                      float& aMax_dK,
                                      float& aMax_dV,
                                      tensor<T>& dQ_val,
                                      tensor<T>& dK_val,
                                      tensor<T>& dV_val)
{

    // Calculate dV_val = softmax_T x dO

    tensor<float> softmax_fp8(softmax_fp32.desc.GetLengths());
    tensor<float> softmax_dot_dO_fp8(dV_val.desc.GetLengths());
    ScaleMult(softmax_fp32, s_scale, softmax_fp8);
    // fp8 matrix multiplication
    Dot_4D_T_4D(softmax_fp8, dO_val, softmax_dot_dO_fp8);

    tensor<float> softmax_dO_fp32(dV_val.desc.GetLengths());
    ScaleMult(softmax_dot_dO_fp8, s_descale * dO_descale, softmax_dO_fp32);

    aMax_dV = AbsoluteMax(softmax_dO_fp32);

    ScaleMult(softmax_dO_fp32, dV_scale, dV_val);

    auto inputLengths = q_val.desc.GetLengths();
    inputLengths[3]   = inputLengths[2]; // NHSD converting to NHSS

    // Calculate dQ_val and dK_val
    // dO x V
    tensor<float> dO_dot_V_tranpose_val(inputLengths);
    tensor<float> dO_dot_V_tranpose_val_fp32(inputLengths);
    // fp8 matrix multiplication
    Dot_4D_4D_T(dO_val, v_val, dO_dot_V_tranpose_val);
    // Descale dO, Descale V
    ScaleMult(dO_dot_V_tranpose_val, dO_descale * v_descale, dO_dot_V_tranpose_val_fp32);

    // dO . O
    auto o_valLengths = O_val.desc.GetLengths();
    o_valLengths[3]   = 1; // NHSD converting to NHS1 for row reduction tensor
    tensor<float> dO_pointwiseMul_O_rrsum_fp32(o_valLengths);

    tensor<float> o_val_fp32(O_val.desc.GetLengths());
    tensor<float> dO_val_fp32(dO_val.desc.GetLengths());
    // Descale dO
    ScaleMult(O_val, O_descale, o_val_fp32);
    // Descale O
    ScaleMult(dO_val, dO_descale, dO_val_fp32);
    tensor<float> dO_pointwiseMul_O_fp32(dO_val.desc.GetLengths());
    PointWiseMultiply(dO_val_fp32, o_val_fp32, dO_pointwiseMul_O_fp32);
    RowReductionSum(dO_pointwiseMul_O_fp32, dO_pointwiseMul_O_rrsum_fp32);

    // Bias Substraction

    tensor<float> bias_sub_fp32(inputLengths);
    BroadCastSub(dO_dot_V_tranpose_val_fp32, dO_pointwiseMul_O_rrsum_fp32, bias_sub_fp32);

    tensor<float> bias_sub_fp32_pm_softmax(inputLengths);
    tensor<T> bias_sub_fp8_pm_softmax(inputLengths);
    PointWiseMultiply(bias_sub_fp32, softmax_fp32, bias_sub_fp32_pm_softmax);
    // AMax_dS
    aMax_dS = AbsoluteMax(bias_sub_fp32_pm_softmax);

    // s_scale to convert to fp8
    ScaleMult(bias_sub_fp32_pm_softmax, ds_scale, bias_sub_fp8_pm_softmax);

    tensor<float> dQ_val_fp32(dQ_val.desc.GetLengths());
    tensor<float> dK_val_fp32(dK_val.desc.GetLengths());
    // fp8 matrix multiplication
    Dot_4D_4D(bias_sub_fp8_pm_softmax, k_val, dQ_val_fp32);
    // fp8 matrix multiplication
    Dot_4D_T_4D(bias_sub_fp8_pm_softmax, q_val, dK_val_fp32);

    // bring it back to fp32
    ScaleMult(dQ_val_fp32, ds_descale * k_descale, dQ_val_fp32);
    ScaleMult(dK_val_fp32, ds_descale * q_descale, dK_val_fp32);

    aMax_dQ = AbsoluteMax(dQ_val_fp32);
    aMax_dK = AbsoluteMax(dK_val_fp32);

    ScaleMult(dQ_val_fp32, dQ_scale, dQ_val);
    ScaleMult(dK_val_fp32, dK_scale, dK_val);
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

template <typename T>
struct ScaledTensor
{
    tensor<T> mTensor;
    float mScale;
    float mDescale;
};

template <typename T, typename... Dims>
ScaledTensor<T> GenScaledTensor(Dims... nhsd)
{
    auto val_scaled = tensor<T>{nhsd...};
    float bias      = prng::gen_A_to_B(-3.0f, 3.0f);
    auto val_full   = tensor<float>{nhsd...}.generate(
        [bias](auto...) { return prng::gen_A_to_B(-2.5f + bias, 2.5f + bias); });
    float scale   = GetF8Scaling(AbsoluteMax(val_full));
    float descale = 1.f / scale;
    ScaleMult(val_full, scale, val_scaled);
    return {val_scaled, scale, descale};
}

template <typename T, typename... Dims>
ScaledTensor<T> GenScaledTensorBackward(Dims... nhsd)
{
    auto val_full = tensor<T>{nhsd...};
    val_full.for_each([&](auto... id) {
        // backward pass is very sensitive to input data due to possible subtraction of
        // similar values and later significant error amplification
        val_full(id...) = prng::gen_descreet_uniform_sign<T>(4, 60);
    });
    float scale   = 0.5f;
    float descale = 1.0f / scale;
    return {val_full, scale, descale};
};

} // namespace cpu
} // namespace test
