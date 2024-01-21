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

using float8 = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;

namespace test {
namespace cpu {


std::vector<CPUMHATestCase> CPUMHAConfigs()
{
    return {
        {// batch_size, sequence_length, num_heads, problem_dimension, drop_out_rate
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
        // mult-head attention begin

        // create Q, K and V
        Dot_3D_3D(word_position, q_weights, q_val);
        Dot_3D_3D(word_position, k_weights, k_val);
        Dot_3D_3D(word_position, v_weights, v_val);
        print4(q_val, "q_val");
        print4(k_val, "k_val");
        print4(v_val, "v_val");
        
        Dot_4D_4D_T(q_val, k_val, q_dot_k_transpose);
        print4(q_dot_k_transpose, "q_dot_k_transpose");

        // // Attention Scale
        // double sqrt_dk = 1.0 / std::sqrt(d_k);
        // std::cout << "sqrt_dk = " << sqrt_dk << std::endl;
        // Scale(q_dot_k_transpose, sqrt_dk);
        // print4(q_dot_k_transpose, "q_dot_k_transpose after scale");

        // AddMask5D_2D(q_dot_k_transpose, mask);
        // print4(q_dot_k_transpose, "q_dot_k_transpose after mask");


        // // *** seperate softmax operation
        // // double fp8_descale = 1.0/100.0;
        // // //   descale Q
        // // F8_T scale_func     = {0, true};
        // // // Scale(q_val, fp8_descale);
        // // // print5(q_val, "before q_val");
        // // Scalef8(q_val, scale_func, fp8_descale);
        // // print5(q_val, "after q_val");
        // // // //   descale K
        // // Scale(k_val, fp8_descale);

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

        print4(q_dot_k_transpose, "softrmaxxx");


        // // descale Q
        // // double scale_s = 1.0;
        // // Scale(q_dot_k_transpose, scale_s);

        // // O = (Q.dot(Kt)).dot(V)
        Dot_4D_4D(q_dot_k_transpose, v_val, atten_heads);
        print4(atten_heads, "atten_heads X value");
    }

    void TearDown() override
    {
        // verify
    }

private:
    void init()
    {
        d_k  = cpu_mha_test_case.problem_dimension / cpu_mha_test_case.num_heads;

        mask = tensor<T>{
            std::vector<int>{cpu_mha_test_case.sequence_length, cpu_mha_test_case.sequence_length}};
        SetupMask(mask);

        word_position = tensor<T>{std::vector<int>{cpu_mha_test_case.batch_size,
                                  cpu_mha_test_case.sequence_length,
                                  cpu_mha_test_case.problem_dimension}};
        
        q_weights = tensor<T>(std::vector<int>{
                                  cpu_mha_test_case.num_heads, cpu_mha_test_case.problem_dimension, d_k});
        k_weights = q_weights;
        v_weights = q_weights;

        q_val = tensor<T>{cpu_mha_test_case.batch_size,
                          cpu_mha_test_case.num_heads,
                          cpu_mha_test_case.sequence_length,
                          d_k};
        k_val = q_val;
        v_val = q_val;

        atten_heads       = tensor<T>{cpu_mha_test_case.batch_size,
                                cpu_mha_test_case.num_heads,
                                cpu_mha_test_case.sequence_length,
                                d_k};
        // concatinate the atten heads d_k => problem dim

        k_transpose = tensor<T>{cpu_mha_test_case.batch_size,
                                cpu_mha_test_case.num_heads,
                                d_k,
                                cpu_mha_test_case.sequence_length};

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
    }
    
    CPUMHATestCase cpu_mha_test_case;
    
    tensor<T> word_position;

    tensor<T> q_weights;
    tensor<T> k_weights;
    tensor<T> v_weights;

    tensor<T> q_val;
    tensor<T> k_val;
    tensor<T> v_val;

    
    tensor<T> k_transpose;
    tensor<T> q_dot_k_transpose;

    tensor<T> mask;
    tensor<T> rrm;
    tensor<T> zinv_tensors;
    size_t d_k;
    
    tensor<T> atten_heads;

    // row reduction max
};

} // namespace cpu
} // namespace test
