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
#include "../driver/tensor_driver.hpp"
#include "cpu_nllloss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/nllloss.hpp>
#include <miopen/miopen.h>

struct NLLLossTestCase
{
    size_t N             = 0;
    size_t C             = 0;
    size_t D1            = 0;
    size_t D2            = 0;
    bool weight_mode     = false;
    int32_t ignore_index = -1;

    std::vector<size_t> input = {N, C, D1, D2};
    friend std::ostream& operator<<(std::ostream& os, const NLLLossTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D1:" << tc.D1 << " D2:" << tc.D2
                  << " weight_mode:" << tc.weight_mode << " ignore_index:" << tc.ignore_index;
    }

    std::vector<size_t> GetInput() const { return input; }
};

inline std::vector<NLLLossTestCase> NLLLossTestConfigs()
{ // dim, dims
    // clang-format off
    return {{16, 21,512,512,false, 255},
            // {64, 21,254,333,false, 255},
            // {64, 21,213,331,false, 255},
            // {64, 21,240,332,false, 255},
            // {64, 21,212,320,false, 255},
            // {64, 21,218,333,false, 255},
            // {64, 21,270,333,false, 255},
            // {64, 21,237,329,false, 255},
            // {64, 21,225,246,false, 255},
            // {64, 21,240,292,false, 255},
            // {64, 21,288,303,false, 255},
            // {64, 21,274,275,false, 255},
            // {64, 21,273,322,false, 255},
            // {64, 21,240,320,false, 255},
            // {64, 21,238,269,false, 255},
            // {64, 21,213,326,false, 255},
            // {64, 21,297,333,false, 255},
            // {64, 21,212,303,false, 255},
            // {64, 21,230,335,false, 255},
            // {64, 21,198,257,false, 255},
            // {64, 21,283,320,false, 255},
            // {64, 21,175,333,false, 255},
            // {64, 21,267,326,false, 255},
            // {32, 21,256,256,false, 255},
            // {55, 21,112,257,false, 255},
            {24, 21,512,512,true, 255}};
    // clang-format on
}

// FORWARD TEST
template <typename T = float>
struct NLLLossTest : public ::testing::TestWithParam<NLLLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        nllloss_config = GetParam();

        ignore_index = nllloss_config.ignore_index;
        weight_mode  = nllloss_config.weight_mode;

        auto in_dim                    = nllloss_config.GetInput();
        std::vector<size_t> target_dim = {in_dim[0], in_dim[2], in_dim[3]};
        std::vector<size_t> weight_dim = {in_dim[1]};
        std::vector<size_t> out_dim    = {in_dim[0], in_dim[2], in_dim[3]};

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-100.0f), static_cast<T>(-1e-2));
        };
        size_t numclass_C     = in_dim[1];
        auto gen_target_value = [numclass_C](auto...) {
            return prng::gen_A_to_B<int32_t>(0, numclass_C - 1);
        };
        auto gen_weight_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10), static_cast<T>(10));
        };
        auto gen_weight_one = [](auto...) { return static_cast<T>(1); };

        input = tensor<T>{in_dim}.generate(gen_input_value);

        target = tensor<int32_t>{target_dim}.generate(gen_target_value);

        if(!weight_mode)
            weight = tensor<T>{weight_dim}.generate(gen_weight_one);
        else
            weight = tensor<T>{weight_dim}.generate(gen_weight_value);

        output = tensor<T>{out_dim};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dim};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        weight_dev = handle.Write(weight.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_nllloss_forward_4d<T>(input, target, weight, ref_output, ignore_index);

        miopenStatus_t status = miopen::NLLLossForward(handle,
                                                       input.desc,
                                                       input_dev.get(),
                                                       target.desc,
                                                       target_dev.get(),
                                                       weight.desc,
                                                       weight_dev.get(),
                                                       output.desc,
                                                       output_dev.get(),
                                                       ignore_index);
        fflush(stdout);

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
    NLLLossTestCase nllloss_config;

    tensor<T> input;
    tensor<int32_t> target;
    tensor<T> weight;
    tensor<T> output;
    tensor<T> ref_output;

    bool weight_mode;
    int32_t ignore_index;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

// BACKWARD TEST
template <typename T = float>
struct NLLLossTestBwd : public ::testing::TestWithParam<NLLLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        nllloss_config = GetParam();

        ignore_index = nllloss_config.ignore_index;
        weight_mode  = nllloss_config.weight_mode;

        auto in_dim                      = nllloss_config.GetInput();
        std::vector<size_t> target_dim   = {in_dim[0], in_dim[2], in_dim[3]};
        std::vector<size_t> weight_dim   = {in_dim[1]};
        std::vector<size_t> out_grad_dim = {in_dim[0], in_dim[2], in_dim[3]};

        size_t numclass_C     = in_dim[1];
        auto gen_target_value = [numclass_C](auto...) {
            return prng::gen_A_to_B<int32_t>(0, numclass_C - 1);
        };
        auto gen_weight_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-10), static_cast<T>(10));
        };
        auto gen_weight_one = [](auto...) { return static_cast<T>(1); };

        auto gen_output_grad_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-100.0f), static_cast<T>(100.0f));
        };

        input_grad = tensor<T>{in_dim};
        std::fill(input_grad.begin(), input_grad.end(), static_cast<T>(0.0f));

        ref_input_grad = tensor<T>{in_dim};
        std::fill(ref_input_grad.begin(), ref_input_grad.end(), static_cast<T>(0.0f));

        target = tensor<int32_t>{target_dim}.generate(gen_target_value);

        if(!weight_mode)
            weight = tensor<T>{weight_dim}.generate(gen_weight_one);
        else
            weight = tensor<T>{weight_dim}.generate(gen_weight_value);

        output_grad = tensor<T>{out_grad_dim}.generate(gen_output_grad_value);

        input_grad_dev  = handle.Write(input_grad.data);
        target_dev      = handle.Write(target.data);
        weight_dev      = handle.Write(weight.data);
        output_grad_dev = handle.Write(output_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        cpu_nllloss_backward_4d<T>(ref_input_grad, target, weight, output_grad, ignore_index);

        miopenStatus_t status = miopen::NLLLossBackward(handle,
                                                        input_grad.desc,
                                                        input_grad_dev.get(),
                                                        target.desc,
                                                        target_dev.get(),
                                                        weight.desc,
                                                        weight_dev.get(),
                                                        output_grad.desc,
                                                        output_grad_dev.get(),
                                                        ignore_index);

        fflush(stdout);

        EXPECT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_input_grad, input_grad);
        EXPECT_TRUE(miopen::range_distance(ref_input_grad) == miopen::range_distance(input_grad));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
    }
    NLLLossTestCase nllloss_config;

    tensor<T> input_grad;
    tensor<T> ref_input_grad;
    tensor<int32_t> target;
    tensor<T> weight;
    tensor<T> output_grad;

    bool weight_mode;
    int32_t ignore_index;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
};
