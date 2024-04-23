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
#define MIOPEN_BETA_API 1
#include "../driver/tensor_driver.hpp"
#include "cpu_cat.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/cat.hpp>
#include <miopen/miopen.h>

struct CatTestCase
{
    size_t dim;
    std::vector<std::vector<int>> inputs;
    friend std::ostream& operator<<(std::ostream& os, const CatTestCase& tc)
    {
        os << " inputs:";
        for(int i = 0; i < tc.inputs.size(); i++)
        {
            auto input = tc.inputs[i];
            if(i != 0)
                os << ",";
            os << input[0];
            for(int j = 1; j < input.size(); j++)
            {
                os << "x" << input[j];
            }
        }
        return os << " dim:" << tc.dim;
    }

    const std::vector<std::vector<int>>& GetInputs() { return inputs; }
};

std::vector<CatTestCase> CatTestConfigs()
{ // dim, dims
    // clang-format off
    return {{1, {{2, 32, 128, 128, 128}, {2, 32, 128, 128, 128}}},                    // 3dunet
            {2, {{32, 16, 31, 64}, {32, 16, 1, 64}}},                                 // bart
            {2, {{32, 16, 114, 64}, {32, 16, 1, 64}}},
            {1, {{16, 256, 32, 32}, {16, 256, 32, 32}, {16, 256, 32, 32},             // deeplabv3m
                 {16, 256, 32, 32}}},
            {1, {{64, 1056, 7, 7}, {64, 48, 7, 7}, {64, 48, 7, 7}, {64, 48, 7, 7},    // densenet
                 {64, 48, 7, 7}, {64, 48, 7, 7}, {64, 48, 7, 7}, {64, 48, 7, 7}}},
            {1, {{65536, 16}, {65536, 351}}},                                         // dlrm
            {2, {{512, 49, 1024}, {512, 49, 1024}}},                                  // gnmt
            {1, {{256, 112, 14, 14}, {256, 288, 14, 14}, {256, 64, 14, 14},           // googlenet
                 {256, 64, 14, 14}}},
            {1, {{256, 256, 7, 7}, {256, 320, 7, 7}, {256, 128, 7, 7},
                 {256, 128, 7, 7}}},
            {0, {{2, 1024, 768}, {2, 1024, 768}, {2, 1024, 768}}},                    // GPT
            {1, {{16, 192000, 1}, {16, 48000, 1}, {16, 12000, 1}, {16, 3000, 1},      // fasterrcnn
                 {16, 780, 1}}},
            {2, {{3990480, 1, 1}, {3990480, 1, 1}, {3990480, 1, 1}, {3990480, 1, 1}}},
            {1, {{64, 320, 8, 8}, {64, 192, 8, 8}, {64, 768, 8, 8}}},                 // inceptionv3
            {1, {{12277440, 1}, {12277440, 1}, {12277440, 1}, {12277440, 1}}},        // maskrcnn
            {1, {{6, 182400, 4}, {6, 45600, 4}, {6, 11400, 4}, {6, 2850, 4},
                 {6, 741, 4}}},
            {0, {{255780}, {255780}, {255780}, {255780}, {255780}, {255780}}},
            {1, {{4, 136800, 91}, {4, 34200, 91}, {4, 8550, 91}, {4, 2223, 91},       // retinanet
                 {4, 630, 91}}},
            {1, {{256, 232, 7, 7}, {256, 232, 7, 7}}},                                // shufflenetv2
            {1, {{256, 116, 14, 14}, {256, 116, 14, 14}}},
            {1, {{2, 192, 80, 80}, {2, 192, 80, 80}}},                                // yolor
            {1, {{16, 320, 20, 20}, {16, 320, 20, 20}}},
            {0, {{4096, 1312}, {4096, 1312}}},                                        // rnnt
            {0, {{9600, 384}, {128, 384}}},                                           // roberta
            {2, {{256, 81, 5776}, {256, 81, 2166}, {256, 81, 600}, {256, 81, 150},    // ssd
                 {256, 81, 36}, {256, 81, 4}}},
            {1, {{16, 256, 32, 32}, {16, 128, 32, 32}, {16, 64, 32, 32},              // stdc
                 {16, 64, 32, 32}}},
            {1, {{1536, 256, 13, 13}, {1536, 256, 13, 13}}},                          // squeezenet
            {1, {{724, 64, 55, 55}, {724, 64, 55, 55}}}};
    // clang-format on
}

template <typename T = float>
struct CatTest : public ::testing::TestWithParam<CatTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        cat_config     = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dim = cat_config.dim;

        auto in_dims = cat_config.GetInputs();
        std::vector<size_t> out_dim{in_dims[0].begin(), in_dims[0].end()};
        out_dim[dim] = 0;
        for(auto in_dim : in_dims)
        {
            inputs.push_back(tensor<T>{in_dim}.generate(gen_value));
            out_dim[dim] += in_dim[dim];
        }

        output = tensor<T>{out_dim};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dim};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(inputs_dev),
                       [&](auto& input) { return handle.Write(input.data); });

        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_cat_forward<T>(inputs, ref_output, dim);
        std::vector<miopen::TensorDescriptor*> inputDescs;
        std::vector<ConstData_t> inputData;

        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(inputDescs),
                       [](auto& input) { return &input.desc; });
        std::transform(inputs_dev.begin(),
                       inputs_dev.end(),
                       std::back_inserter(inputData),
                       [](auto& input_dev) { return input_dev.get(); });

        miopenStatus_t status = miopen::CatForward(handle,
                                                   inputDescs.size(),
                                                   inputDescs.data(),
                                                   inputData.data(),
                                                   output.desc,
                                                   output_dev.get(),
                                                   dim);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        auto error = miopen::rms_range(ref_output, output);
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error == 0) << "Outputs do not match each other. Error:" << error;
    }
    CatTestCase cat_config;

    std::vector<tensor<T>> inputs;
    tensor<T> output;
    tensor<T> ref_output;

    std::vector<miopen::Allocator::ManageDataPtr> inputs_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    size_t dim;
};
