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

namespace test {
namespace cpu {

std::vector<CPUMHATestCase> CPUMHAConfigs()
{
    // clang-format off
    return {{// batch_size, sequence_length, num_heads, problem_dimension, drop_out_rate
                  2,             5,             2,           4,                0.0}};
    // clang-format on
}

template <typename T = float>
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

        double sqr_dk = std::sqrt(q_val.desc.GetLengths()[3]);
        ScaleMult(q_val, 1.0 / sqr_dk, q_val);

        MultiHeadAttentionf32(
            q_val, k_val, v_val, q_dot_k_transpose, softmax, attn_max, z_sum, multi_head_attention);

        Concat(multi_head_attention, concatinated_attention);
        Dot_3D_2D_T(
            concatinated_attention, final_linear_transform_weights, final_transformed_attention);

        MultiHeadAttentionfp8(q_val,
                              k_val,
                              v_val,
                              q_dot_k_transpose,
                              attn_max,
                              q_scale,
                              k_scale,
                              aMax_S,
                              s_scale,
                              v_scale,
                              scale_O,
                              multi_head_attention_fp8);
        Concat(multi_head_attention_fp8, concatinated_attention_fp8);

        MultiHeadAttentionBackwardDataf32(q_val,
                                          k_val,
                                          v_val,
                                          multi_head_attention, // o_val
                                          dO_val,
                                          q_dot_k_transpose,
                                          softmax,
                                          attn_max,
                                          z_sum,
                                          dQ_val,
                                          dK_val,
                                          dV_val);
    }

    void TearDown() override
    {
        tensor<T> attention_golden(final_transformed_attention.desc.GetLengths());
        ExtractGoldenDataFromJson(json_attention_golden_data_fwd, attention_golden);

        double error     = miopen::rms_range(attention_golden, final_transformed_attention);
        double threshold = 1e-5;
        EXPECT_TRUE(error < threshold)
            << "Error beyond tolerance Error:" << error << ",  Threshold: " << threshold;
    }

    void init()
    {
        d_k = cpu_mha_test_case.problem_dimension / cpu_mha_test_case.num_heads;

        // mask = tensor<T>{
        //     std::vector<int>{cpu_mha_test_case.sequence_length,
        //     cpu_mha_test_case.sequence_length}};
        // SetupMask(mask);

        word_position = tensor<T>{std::vector<int>{cpu_mha_test_case.batch_size,
                                                   cpu_mha_test_case.sequence_length,
                                                   cpu_mha_test_case.problem_dimension}};

        // since Pytorch's Y = X*W_tranpose
        // cpu_mha_test_case.num_heads, cpu_mha_test_case.problem_dimension, d_k
        //          need to change the dimension to
        // cpu_mha_test_case.num_heads, d_k, cpu_mha_test_case.problem_dimension

        q_weights = tensor<T>(std::vector<int>{
            cpu_mha_test_case.num_heads, d_k, cpu_mha_test_case.problem_dimension});
        k_weights = q_weights;
        v_weights = q_weights;

        q_val = tensor<T>{cpu_mha_test_case.batch_size,
                          cpu_mha_test_case.num_heads,
                          cpu_mha_test_case.sequence_length,
                          d_k};
        k_val = q_val;
        v_val = q_val;

        multi_head_attention           = tensor<T>{cpu_mha_test_case.batch_size,
                                         cpu_mha_test_case.num_heads,
                                         cpu_mha_test_case.sequence_length,
                                         d_k};
        final_linear_transform_weights = tensor<T>(std::vector<int>{
            cpu_mha_test_case.problem_dimension, cpu_mha_test_case.problem_dimension});

        concatinated_attention      = tensor<T>{std::vector<int>{
            cpu_mha_test_case.batch_size,
            cpu_mha_test_case.sequence_length,
            cpu_mha_test_case.problem_dimension}}; // cpu_mha_test_case.num_heads*d_k
        final_transformed_attention = concatinated_attention;

        concatinated_attention_fp8 =
            tensor<float8>{std::vector<int>{cpu_mha_test_case.batch_size,
                                            cpu_mha_test_case.sequence_length,
                                            cpu_mha_test_case.problem_dimension}};
        multi_head_attention_fp8 = tensor<float8>{cpu_mha_test_case.batch_size,
                                                  cpu_mha_test_case.num_heads,
                                                  cpu_mha_test_case.sequence_length,
                                                  d_k};

        q_dot_k_transpose = tensor<T>{cpu_mha_test_case.batch_size,
                                      cpu_mha_test_case.num_heads,
                                      cpu_mha_test_case.sequence_length,
                                      cpu_mha_test_case.sequence_length};
        softmax           = q_dot_k_transpose;
        // reduce row max
        attn_max = tensor<T>{cpu_mha_test_case.batch_size,
                             cpu_mha_test_case.num_heads,
                             cpu_mha_test_case.sequence_length,
                             1};
        z_sum    = attn_max;

        word_position.generate(GenData<T>{});
        q_weights.generate(GenData<T>{});
        k_weights.generate(GenData<T>{});
        v_weights.generate(GenData<T>{});

        final_linear_transform_weights.generate(GenData<T>{});

        // backward
        dO_val = v_val;
        dQ_val = dO_val;
        dK_val = dO_val;
        dV_val = dO_val;
        dO_val.generate(GenData<T>{});
        // backward
    }

    CPUMHATestCase cpu_mha_test_case;

    // input
    tensor<T> word_position;

    size_t d_k;

    // weights
    tensor<T> q_weights;
    tensor<T> k_weights;
    tensor<T> v_weights;
    // This for the final linear transformation
    // of the attention.
    tensor<T> final_linear_transform_weights;

    // QKV vectors
    tensor<T> q_val;
    tensor<T> k_val;
    tensor<T> v_val;

    // softmax
    tensor<T> q_dot_k_transpose;
    tensor<T> attn_max;
    tensor<T> z_sum;
    tensor<T> softmax;

    // attention
    tensor<T> multi_head_attention;
    tensor<float8> multi_head_attention_fp8;

    tensor<T> concatinated_attention;
    tensor<float8> concatinated_attention_fp8;

    tensor<T> final_transformed_attention;

    // scales
    double q_scale;
    double k_scale;
    double aMax_S;
    double s_scale;
    double v_scale;
    double scale_O;

    // backward
    tensor<T> dQ_val;
    tensor<T> dK_val;
    tensor<T> dV_val;
    tensor<T> dO_val;
    tensor<T> o_val;
};

} // namespace cpu
} // namespace test
