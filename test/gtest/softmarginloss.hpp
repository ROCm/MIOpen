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
#include <numeric>

struct SoftMarginLossUnreducedTestCase
{
    std::vector<size_t> dims;
    std::vector<size_t> strides;
};

std::vector<SoftMarginLossUnreducedTestCase> SoftMarginLossTestConfigs()
{
    // clang-format off
    return {
    {{256, 4, 8732}, {34928, 8732, 1}}, // squeezenet
    {{32, 80, 870}, {69600, 1, 80}}, // t5
    {{32, 80, 870}, {69600, 870, 1}}, // t5
    {{4, 182403, 91}, {16598673, 91, 1}}, // resnext
    {{1534680}, {1}}, // maskrcnn
    {{16, 1, 512, 512}, {262144, 262144, 512, 1}}, // stdc
    {{2, 3, 160, 160}, {6528000, 2176000, 13600, 85}}, // yolor
    {{2, 3, 80, 80}, {1632000, 544000, 6800, 85}}, // yolor
    {{32756, 80}, {85, 1}}, // yolov5
    {{64, 3, 80, 80}, {1632000, 544000, 6800, 85}}, // yolov5
    {{64, 3, 40, 40}, {408000, 136000, 3400, 85}}, // yolov5
    {{22311, 80}, {85, 1}}, // yolov5
    {{64, 3, 20, 20}, {102000, 34000, 1700, 85}}, // yolov5
    {{8, 4}, {4, 1}}, // ssd
    {{56, 4}, {4, 1}}, // ssd
    {{131, 4}, {4, 1}}, // ssd
    {{10000}, {1}}, // 1dcont
    {{200, 50}, {50, 1}}, // 2dcont
    {{20, 50, 10}, {500, 10, 1}}, // 3dcont
    {{4, 25, 4, 25}, {2500, 100, 25, 1}}, // 4dcont
    {{12, 3, 4, 5, 6}, {360, 120, 30, 6, 1}}, // 5dcont
    {{10000}, {3}}, // 1d-uncont
    {{200, 50}, {1, 200}}, // 2d-uncont
    {{200, 50}, {505, 1}}, // 2d-unpacked
    {{20, 50, 10}, {1, 20, 1000}}, // 3d-uncont
    {{20, 50, 10}, {7575, 15, 1}}, // 3d-unpacked
    {{4, 25, 4, 25}, {1, 16, 4, 400}}, // 4d-uncont
    {{4, 25, 4, 25}, {5859, 217, 31, 1}}, // 4d-unpacked
    {{12, 3, 4, 5, 6}, {360, 120, 6, 24, 1}}, // 5d-uncont
    {{12, 3, 4, 5, 6}, {5760, 960, 120, 12, 1}}, // 5d-unpacked
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

        cpu_softmarginloss_unreduced_forward<T>(input, target, ref_output);
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

template <typename T = float>
struct SoftMarginLossReducedForwardTest
    : public ::testing::TestWithParam<SoftMarginLossUnreducedTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                  = get_handle();
        softmarginlossunreduced_config = GetParam();

        auto in_dims    = softmarginlossunreduced_config.dims;
        auto in_strides = softmarginlossunreduced_config.strides;

        auto input_numel =
            std::accumulate(in_dims.begin(), in_dims.end(), 1L, std::multiplies<int64_t>());
        if(std::is_same<T, half_float::half>::value && input_numel > 80000)
        {
            std::cerr << "For fp16 test, too many elements in input tensor can lead to fp16 "
                         "overflow when doing reduction"
                      << std::endl;
            GTEST_SKIP();
        }

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

        // Tensor with 1 element to store result after reduce
        output = tensor<T>{std::vector<size_t>{1}};
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<T>{std::vector<size_t>{1}};
        std::fill(ref_output.begin(), ref_output.end(), 0);

        ws_sizeInBytes = miopen::GetSoftMarginLossForwardWorkspaceSize(
            handle, input.desc, target.desc, output.desc, 1);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        std::vector<size_t> workspace_dims;
        workspace_dims.push_back(ws_sizeInBytes / sizeof(T));

        workspace = tensor<T>{workspace_dims};
        std::fill(workspace.begin(), workspace.end(), 0);

        ref_workspace = tensor<T>{workspace_dims};
        std::fill(ref_workspace.begin(), ref_workspace.end(), 0);

        // Write from CPU to GPU
        input_dev     = handle.Write(input.data);
        target_dev    = handle.Write(target.data);
        output_dev    = handle.Write(output.data);
        workspace_dev = handle.Write(workspace.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        // Mean reduction. To test with sum reduction, change divisor to 1
        float divisor = input.desc.GetElementSize();
        cpu_softmarginloss_reduced_forward<T>(input, target, ref_output, ref_workspace, divisor);

        miopenStatus_t status;
        status = miopen::SoftMarginLossForward(handle,
                                               workspace_dev.get(),
                                               ws_sizeInBytes,
                                               input.desc,
                                               input_dev.get(),
                                               target.desc,
                                               target_dev.get(),
                                               output.desc,
                                               output_dev.get(),
                                               divisor);
        EXPECT_EQ(status, miopenStatusSuccess);

        // Write from GPU to CPU
        workspace.data = handle.Read<T>(workspace_dev, workspace.data.size());
        output.data    = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        // fp32: 1.19209e-07, fp16: 0.000976562, bf16: 0.0078125
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);
        std::cerr << "Error: " << error << std::endl;
        std::cerr << "Threshold: " << threshold << std::endl;
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance "
                                               "Error:"
                                            << error << ",  Threshold x 10: " << threshold * 10;
    }
    SoftMarginLossUnreducedTestCase softmarginlossunreduced_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<T> workspace;

    tensor<T> ref_output;
    tensor<T> ref_workspace;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};
