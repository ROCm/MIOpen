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
#include "cpu_softmarginloss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/softmarginloss.hpp>

struct SoftMarginLossUnreducedTestCase
{
    std::vector<size_t> dims;
    std::vector<size_t> strides;
};
std::vector<SoftMarginLossUnreducedTestCase> SoftMarginLossUnreducedTestConfigs()
{
    // clang-format off
    return {
        {{1, 4}, {4, 1}}, // ssd (small test)
        {{12, 4}, {4, 1}}, // ssd
        {{10}, {1}}, // 1d cont
        {{3, 4}, {4, 1}}, // 2d cont
        {{2, 3, 4}, {12, 4, 1}}, // 3d cont
        {{2, 3, 4, 5}, {60, 20, 5 ,1}}, // 4d cont
        {{2, 3, 4, 5, 6}, {360, 120, 30, 6, 1}}, // 5d cont
        {{256, 4, 8732}, {34928, 8732, 1}}, // squeezenet (3d cont)
        {{32, 80, 870}, {69600, 870, 1}}, // t5
        {{4, 182403, 91}, {16598673, 91, 1}}, // resnext (big test >= 1M elements)
        {{1534680}, {1}}, // maskrcnn (1d cont)
        {{16, 1, 512, 512}, {262144, 262144, 512, 1}}, // stdc (4d cont)
        {{32, 80, 870}, {69600, 1, 80}}, // t5 (3d uncontiguous, packed)
        {{2, 3, 160, 160}, {6528000, 2176000, 13600, 85}}, // yolor (4d uncontiguous, unpacked)
        {{32756, 80}, {85, 1}}, // yolov5 (2d uncontiguous, unpacked)
        {{64, 3, 80, 80}, {1632000, 544000, 6800, 85}}, // yolov5 (4d uncontiguous, unpacked)
    };
    // clang-format on
}

template <typename T = float>
struct SoftMarginLossUnreducedForwardTest
    : public ::testing::TestWithParam<SoftMarginLossUnreducedTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                  = get_handle();
        softmarginlossunreduced_config = GetParam();

        auto in_dims      = softmarginlossunreduced_config.dims;
        auto in_strides   = softmarginlossunreduced_config.strides;
        auto gen_in_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        // below commented code that I have seen in many files will not work correctly with unpacked
        // tensor tensor.generate() will call for_each() and this function only iterate through
        // desc.GetLengths().size(), not desc.GetElementSpace() Example input_tensor to verify:
        // input_tensor: dim(5, 3), stride(4, 1). Element space = 19, size = 15. Call above code
        // will only generate value for 15 elements

        // input = tensor<T>{in_dims, in_strides}.generate(gen_in_value);

        // This is the right method to generate value for tensor
        input = tensor<T>{in_dims, in_strides};
        std::generate(input.begin(), input.end(), gen_in_value);

        auto gen_target_value = [](auto...) {
            return (prng::gen_A_to_B<int32_t>(0, 2) == 0) ? -1 : 1;
        };
        target = tensor<T>{in_dims, in_strides};
        std::generate(target.begin(), target.end(), gen_target_value);

        output = tensor<T>{in_dims, in_strides};
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<T>{in_dims, in_strides};
        std::fill(ref_output.begin(), ref_output.end(), 0);

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_softmarginloss_unreduced_forward<T>(input, ref_output, target);
        miopenStatus_t status;

        status = miopen::SoftMarginLossUnreducedForward(handle,
                                                        input.desc,
                                                        input_dev.get(),
                                                        target.desc,
                                                        target_dev.get(),
                                                        output.desc,
                                                        output_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
        // std::cerr << "Input: ";
        // for(int i = 0; i < input.data.size(); i++)
        //     std::cerr << input[i] << " ";
        // std::cerr << "\nTarget: ";
        // for(int i = 0; i < target.data.size(); i++)
        //     std::cerr << target[i] << " ";
        // std::cerr << "\nOutput: ";
        // for(int i = 0; i < output.data.size(); i++)
        //     std::cerr << output[i] << " ";
        // std::cerr << "\nRef_output: ";
        // for(int i = 0; i < ref_output.data.size(); i++)
        //     std::cerr << ref_output[i] << " ";
        // std::cerr << std::endl;
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold) << "Error output beyond tolerance "
                                          "Error:"
                                       << error << ",  Threshold: " << threshold;
    }
    SoftMarginLossUnreducedTestCase softmarginlossunreduced_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T = float>
struct SoftMarginLossUnreducedBackwardTest
    : public ::testing::TestWithParam<SoftMarginLossUnreducedTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                  = get_handle();
        softmarginlossunreduced_config = GetParam();

        auto in_dims      = softmarginlossunreduced_config.dims;
        auto in_strides   = softmarginlossunreduced_config.strides;
        auto gen_in_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        // Contiguous or not? depends on strides
        input = tensor<T>{in_dims, in_strides}.generate(gen_in_value);

        auto gen_target_value = [](auto...) {
            return (prng::gen_A_to_B<int32_t>(0, 2) == 0) ? -1 : 1;
        };
        target = tensor<T>{in_dims, in_strides}.generate(gen_target_value);

        dO = tensor<T>{in_dims, in_strides};
        std::fill(dO.begin(), dO.end(), 1);

        dI = tensor<T>{in_dims, in_strides};
        std::fill(dI.begin(), dI.end(), 0);

        ref_dI = tensor<T>{in_dims, in_strides};
        std::fill(ref_dI.begin(), ref_dI.end(), 0);

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        dO_dev     = handle.Write(dO.data);
        dI_dev     = handle.Write(dI.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_softmarginloss_unreduced_backward<T>(input, target, dO, ref_dI);
        miopenStatus_t status;

        status = miopen::SoftMarginLossUnreducedBackward(handle,
                                                         input.desc,
                                                         input_dev.get(),
                                                         target.desc,
                                                         target_dev.get(),
                                                         dO.desc,
                                                         dO_dev.get(),
                                                         dI.desc,
                                                         dI_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        dI.data = handle.Read<T>(dI_dev, dI.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_dI, dI);
        EXPECT_TRUE(miopen::range_distance(ref_dI) == miopen::range_distance(dI));
        EXPECT_TRUE(error < threshold) << "Error output beyond tolerance "
                                          "Error:"
                                       << error << ",  Threshold: " << threshold;
    }
    SoftMarginLossUnreducedTestCase softmarginlossunreduced_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> dO;
    tensor<T> dI;

    tensor<T> ref_dI;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dO_dev;
    miopen::Allocator::ManageDataPtr dI_dev;
};
