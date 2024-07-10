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

#include "../driver/tensor_driver.hpp"
#include "cpu_outer.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/outer.hpp>

struct OuterTestCase
{
    size_t N;
    size_t M;
    friend std::ostream& operator<<(std::ostream& os, const OuterTestCase& tc)
    {
        return os << " N:" << tc.N << " M:" << tc.M;
    }

    std::vector<size_t> GetInput() { return std::vector<size_t>({N, M}); }
};

std::vector<OuterTestCase> OuterTestConfigs()
{ // n c d h w dim nanPropagation
    // clang-format off
    return {
        {128,128},
        {128,2048},
        {128,32768},
        {2048,128},
        {2048,2048},
        {2048,32768},
        {32768,128},
        {32768,2048},
        {32768,32768}
    };
    // clang-format on
}

template <typename T = float>
struct OuterFwdTest : public ::testing::TestWithParam<OuterTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        outer_config   = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 10); };

        auto in_dims = outer_config.GetInput();

        input1 = tensor<T>{std::vector<size_t>({in_dims[0]})}.generate(gen_value);
        input2 = tensor<T>{std::vector<size_t>({in_dims[1]})}.generate(gen_value);

        std::vector<size_t> out_dims;

        for(int i = 0; i < in_dims.size(); i++)
        {
            out_dims.push_back(in_dims[i]);
        }

        output = tensor<T>{out_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input1_dev = handle.Write(input1.data);
        input2_dev = handle.Write(input2.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_outer_forward<T>(input1, input2, ref_output);
        miopenStatus_t status;

        status = miopen::OuterForward(handle,
                                      input1.desc,
                                      input1_dev.get(),
                                      input2.desc,
                                      input2_dev.get(),
                                      output.desc,
                                      output_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    OuterTestCase outer_config;

    tensor<T> input1;
    tensor<T> input2;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input1_dev;
    miopen::Allocator::ManageDataPtr input2_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T = float>
struct OuterBwdTest : public ::testing::TestWithParam<OuterTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        outer_config   = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1, 10); };

        auto in_dims = outer_config.GetInput();

        input1 = tensor<T>{std::vector<size_t>({in_dims[0]})}.generate(gen_value);
        input2 = tensor<T>{std::vector<size_t>({in_dims[1]})}.generate(gen_value);

        std::vector<size_t> out_dims;
        for(int i = 0; i < in_dims.size(); i++)
        {
            out_dims.push_back(in_dims[i]);
        }
        outputGrad = tensor<T>{out_dims}.generate(gen_value);

        input1Grad = tensor<T>{std::vector<size_t>({in_dims[0]})};
        input2Grad = tensor<T>{std::vector<size_t>({in_dims[1]})};

        std::fill(input1Grad.begin(), input1Grad.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(input2Grad.begin(), input2Grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input1Grad = tensor<T>{std::vector<size_t>({in_dims[0]})};
        ref_input2Grad = tensor<T>{std::vector<size_t>({in_dims[1]})};

        input1_dev     = handle.Write(input1.data);
        input2_dev     = handle.Write(input2.data);
        outputGrad_dev = handle.Write(outputGrad.data);

        input1Grad_dev = handle.Write(input1Grad.data);
        input2Grad_dev = handle.Write(input2Grad.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_outer_backward<T>(input1, input2, outputGrad, ref_input1Grad, ref_input2Grad);
        miopenStatus_t status;

        status = miopen::OuterBackwardGrad1(handle,
                                            input2.desc,
                                            input2_dev.get(),
                                            input1Grad.desc,
                                            input1Grad_dev.get(),
                                            outputGrad.desc,
                                            outputGrad_dev.get());

        status = miopen::OuterBackwardGrad2(handle,
                                            input1.desc,
                                            input1_dev.get(),
                                            input2Grad.desc,
                                            input2Grad_dev.get(),
                                            outputGrad.desc,
                                            outputGrad_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        input1Grad.data = handle.Read<T>(input1Grad_dev, input1Grad.data.size());
        input2Grad.data = handle.Read<T>(input2Grad_dev, input2Grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error1      = miopen::rms_range(ref_input1Grad, input1Grad);
        auto error2      = miopen::rms_range(ref_input1Grad, input1Grad);

        EXPECT_TRUE(miopen::range_distance(ref_input1Grad) == miopen::range_distance(input1Grad));
        EXPECT_TRUE(miopen::range_distance(ref_input2Grad) == miopen::range_distance(input2Grad));
        EXPECT_TRUE(error1 < threshold * 10) << "Error1 output beyond tolerance Error:" << error1
                                             << ",  Thresholdx10: " << threshold * 10;
        EXPECT_TRUE(error2 < threshold * 10) << "Error2 output beyond tolerance Error:" << error2
                                             << ",  Thresholdx10: " << threshold * 10;
    }
    OuterTestCase outer_config;

    tensor<T> input1;
    tensor<T> input2;
    tensor<T> input1Grad;
    tensor<T> input2Grad;
    tensor<T> outputGrad;

    tensor<T> ref_input1Grad;
    tensor<T> ref_input2Grad;

    miopen::Allocator::ManageDataPtr input1_dev;
    miopen::Allocator::ManageDataPtr input2_dev;
    miopen::Allocator::ManageDataPtr input1Grad_dev;
    miopen::Allocator::ManageDataPtr input2Grad_dev;
    miopen::Allocator::ManageDataPtr outputGrad_dev;
};