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

#include "mha_helper.hpp"
#include "attention_golden.hpp"
#include "verify.hpp"

namespace test {
namespace cpu {

std::vector<CPUMHATestCase> CPUMHAConfigs()
{
    // clang-format off
    return {{// batch_size, sequence_length, num_heads, problem_dimension, drop_out_rate
                  2,             5,             2,           4,                0.0f}};
    // clang-format on
}

template <typename InputType, typename OutputType>
struct CPUMHATest : public ::testing::TestWithParam<CPUMHATestCase>
{
protected:
    void SetUp() override
    {
        cpu_mha_test_case = GetParam();

        // Initialize the tensors
        init();
        // fp32 Q, K, V
        Dot_3D_3D_T(word_position, q_weights, q_val);
        Dot_3D_3D_T(word_position, k_weights, k_val);
        Dot_3D_3D_T(word_position, v_weights, v_val);

        float sqr_dk = std::sqrt(q_val.desc.GetLengths()[3]);
        ScaleMult(q_val, 1.0f / sqr_dk, q_val);

        if constexpr(std::is_same_v<OutputType, float>)
        {
            MultiHeadAttentionf32(q_val,
                                  k_val,
                                  v_val,
                                  q_dot_k_transpose,
                                  softmax,
                                  attn_max,
                                  z_sum,
                                  aMax_S,
                                  aMax_O,
                                  multi_head_attention);

            Concat(multi_head_attention, concatinated_attention);
        }
        else
        {
            float q_scale = GetF8Scaling(FindMax4D(q_val));
            float k_scale = GetF8Scaling(FindMax4D(k_val));
            float v_scale = GetF8Scaling(FindMax4D(v_val));
            float q_descale = 1.f / q_scale;
            float k_descale = 1.f / k_scale;
            float v_descale = 1.f / v_scale;

            float s_scale = 1.f;
            float s_descale = 1.f / s_scale;

            float o_scale = 1.f;
            float o_descale = 1.f / o_scale;

            tensor<float8> q_val_fp8(q_val.desc.GetLengths());
            tensor<float8> k_val_fp8(k_val.desc.GetLengths());
            tensor<float8> v_val_fp8(v_val.desc.GetLengths());

            ScaleMult(q_val, q_scale, q_val_fp8);
            ScaleMult(k_val, k_scale, k_val_fp8);
            ScaleMult(v_val, v_scale, v_val_fp8);

            MultiHeadAttentionfp8(q_val_fp8,
                                  k_val_fp8,
                                  v_val_fp8,
                                  attn_max,
                                  z_sum,
                                  q_descale,
                                  k_descale,
                                  v_descale,
                                  s_descale,
                                  s_scale,
                                  o_scale,
                                  aMax_S,
                                  aMax_O,
                                  multi_head_attention);
            Concat(multi_head_attention, final_transformed_attention);
            ScaleMult(final_transformed_attention, o_descale, concatinated_attention);
        }

        Dot_3D_2D_T(
            concatinated_attention, final_linear_transform_weights, final_transformed_attention);
    }

    void TearDown() override
    {
        auto calcStats =
            [ref = ExtractGoldenDataFromJson(json_attention_golden_data,
                                             final_transformed_attention)](const auto& tensor_val) {
                return std::tuple{miopen::rms_range(ref, tensor_val),
                                  miopen::max_diff(ref, tensor_val)};
            };

        const auto [error, max_diff] = calcStats(final_transformed_attention);
        // CI clang-tidy treats is as "boolean value assigned to float"
        const double error_threshold    = ((std::is_same_v<OutputType, float>) ? 1e-7 : 5e-2);
        const double max_diff_threshold = ((std::is_same_v<OutputType, float>) ? 1e-6 : 5e-1);

        EXPECT_LT(error, error_threshold);
        EXPECT_LT(max_diff, max_diff_threshold);
    }

    void init()
    {
        d_k = cpu_mha_test_case.problem_dimension / cpu_mha_test_case.num_heads;

        // mask = tensor<T>{
        //     std::vector<int>{cpu_mha_test_case.sequence_length,
        //     cpu_mha_test_case.sequence_length}};
        // SetupMask(mask);

        word_position = tensor<InputType>{std::vector<int>{cpu_mha_test_case.batch_size,
                                                           cpu_mha_test_case.sequence_length,
                                                           cpu_mha_test_case.problem_dimension}};

        // since Pytorch's Y = X*W_tranpose
        // cpu_mha_test_case.num_heads, cpu_mha_test_case.problem_dimension, d_k
        //          need to change the dimension to
        // cpu_mha_test_case.num_heads, d_k, cpu_mha_test_case.problem_dimension

        q_weights = tensor<InputType>(std::vector<int>{cpu_mha_test_case.num_heads,           //
                                                       d_k,                                   //
                                                       cpu_mha_test_case.problem_dimension}); //
        k_weights = q_weights;
        v_weights = q_weights;

        q_val = tensor<InputType>{cpu_mha_test_case.batch_size,
                                  cpu_mha_test_case.num_heads,
                                  cpu_mha_test_case.sequence_length,
                                  d_k};
        k_val = q_val;
        v_val = q_val;

        multi_head_attention = tensor<OutputType>{cpu_mha_test_case.batch_size,
                                                  cpu_mha_test_case.num_heads,
                                                  cpu_mha_test_case.sequence_length,
                                                  d_k};

        final_linear_transform_weights =
            tensor<InputType>(std::vector<int>{cpu_mha_test_case.problem_dimension,   //
                                               cpu_mha_test_case.problem_dimension}); //

        concatinated_attention      = tensor<InputType>{std::vector<int>{
            cpu_mha_test_case.batch_size,
            cpu_mha_test_case.sequence_length,
            cpu_mha_test_case.problem_dimension}}; // cpu_mha_test_case.num_heads*d_k
        final_transformed_attention = concatinated_attention;

        q_dot_k_transpose = tensor<float>{cpu_mha_test_case.batch_size,
                                          cpu_mha_test_case.num_heads,
                                          cpu_mha_test_case.sequence_length,
                                          cpu_mha_test_case.sequence_length};
        softmax           = q_dot_k_transpose;
        // reduce row max
        attn_max = tensor<float>{cpu_mha_test_case.batch_size,
                                 cpu_mha_test_case.num_heads,
                                 cpu_mha_test_case.sequence_length,
                                 1};
        z_sum    = attn_max;

        word_position = ExtractGoldenDataFromJson(json_attention_word_position, word_position);
        q_weights     = ExtractGoldenDataFromJson(json_attention_q_weights, q_weights);
        k_weights     = ExtractGoldenDataFromJson(json_attention_k_weights, k_weights);
        v_weights     = ExtractGoldenDataFromJson(json_attention_v_weights, v_weights);

        final_linear_transform_weights = ExtractGoldenDataFromJson(
            json_attention_final_linear_transform_weights, final_linear_transform_weights);
    }

    CPUMHATestCase cpu_mha_test_case;

    // input
    tensor<InputType> word_position;

    size_t d_k;

    // weights
    tensor<InputType> q_weights;
    tensor<InputType> k_weights;
    tensor<InputType> v_weights;
    // This for the final linear transformation
    // of the attention.
    tensor<InputType> final_linear_transform_weights;

    // QKV vectors
    tensor<InputType> q_val;
    tensor<InputType> k_val;
    tensor<InputType> v_val;

    // softmax
    tensor<float> q_dot_k_transpose;
    tensor<float> attn_max;
    tensor<float> z_sum;
    tensor<float> softmax;

    // attention
    tensor<OutputType> multi_head_attention;

    tensor<InputType> concatinated_attention;
    tensor<InputType> final_transformed_attention;

    // scales
    float aMax_S;
    float aMax_O;
};

} // namespace cpu
} // namespace test
