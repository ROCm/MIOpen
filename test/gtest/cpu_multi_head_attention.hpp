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

namespace test {
namespace cpu {

std::vector<CPUMHATestCase> CPUMHAConfigs()
{
    return {{// batch_size, sequence_length, num_heads, problem_dimension, drop_out_rate
             2,
             5,
             2,
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

        // Initialize the tensors
        init();
        // fp32 Q, K, V
        Dot_3D_3D_T(word_position, q_weights, q_val);
        Dot_3D_3D_T(word_position, k_weights, k_val);
        Dot_3D_3D_T(word_position, v_weights, v_val);

        double sqr_dk = std::sqrt(q_val.desc.GetLengths()[3]);
        ScaleToFP32(q_val, sqr_dk);

        MultiHeadAttentionf32(
            q_val, k_val, v_val, q_dot_k_transpose, rrm, zinv_tensors, atten_heads);

        Concat(atten_heads, concatinated_O_val);
        Dot_3D_2D_T(concatinated_O_val, final_linear_transform_weights, final_atten_heads);

        MultiHeadAttentionfp8(
            q_val, k_val, v_val, q_dot_k_transpose, rrm, zinv_tensors, atten_heads_fp8);
        Concat(atten_heads_fp8, concatinated_O_val_fp8);
    }

    void TearDown() override
    {
        tensor<T> attention_golden(final_atten_heads.desc.GetLengths());
        ExtractGoldenDataFromJson("../test/gtest/attention_golden.json", attention_golden);
        output_tensor_to_screen(q_val, "q_val", 4);

        double error     = miopen::rms_range(attention_golden, final_atten_heads);
        double threshold = 0.155;
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

        atten_heads                    = tensor<T>{cpu_mha_test_case.batch_size,
                                cpu_mha_test_case.num_heads,
                                cpu_mha_test_case.sequence_length,
                                d_k};
        final_linear_transform_weights = tensor<T>(std::vector<int>{
            cpu_mha_test_case.problem_dimension, cpu_mha_test_case.problem_dimension});

        concatinated_O_val = tensor<T>{std::vector<int>{
            cpu_mha_test_case.batch_size,
            cpu_mha_test_case.sequence_length,
            cpu_mha_test_case.problem_dimension}}; // cpu_mha_test_case.num_heads*d_k
        final_atten_heads  = concatinated_O_val;

        concatinated_O_val_fp8 =
            tensor<float8>{std::vector<int>{cpu_mha_test_case.batch_size,
                                            cpu_mha_test_case.sequence_length,
                                            cpu_mha_test_case.problem_dimension}};
        atten_heads_fp8 = tensor<float8>{cpu_mha_test_case.batch_size,
                                         cpu_mha_test_case.num_heads,
                                         cpu_mha_test_case.sequence_length,
                                         d_k};

        q_dot_k_transpose = tensor<T>{cpu_mha_test_case.batch_size,
                                      cpu_mha_test_case.num_heads,
                                      cpu_mha_test_case.sequence_length,
                                      cpu_mha_test_case.sequence_length};
        // reduce row max
        rrm = tensor<T>{cpu_mha_test_case.batch_size,
                        cpu_mha_test_case.num_heads,
                        cpu_mha_test_case.sequence_length,
                        1};
        //
        zinv_tensors = tensor<T>{cpu_mha_test_case.batch_size,
                                 cpu_mha_test_case.num_heads,
                                 cpu_mha_test_case.sequence_length,
                                 1};

        word_position.generate(GenData<T>{});
        q_weights.generate(GenData<T>{});
        k_weights.generate(GenData<T>{});
        v_weights.generate(GenData<T>{});

        final_linear_transform_weights.generate(GenData<T>{});
    }

    CPUMHATestCase cpu_mha_test_case;

    tensor<T> word_position;

    tensor<T> q_weights;
    tensor<T> k_weights;
    tensor<T> v_weights;
    tensor<T> final_linear_transform_weights;

    tensor<T> q_val;
    tensor<T> k_val;
    tensor<T> v_val;

    tensor<T> q_dot_k_transpose;
    tensor<T> concatinated_O_val;

    // tensor<T> mask;
    tensor<T> rrm;
    tensor<T> zinv_tensors;
    size_t d_k;

    tensor<T> atten_heads;
    tensor<T> final_atten_heads;

    tensor<float8> concatinated_O_val_fp8;
    tensor<float8> atten_heads_fp8;

    // row reduction max
};

} // namespace cpu
} // namespace test
