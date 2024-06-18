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
#include "cpu_multilabelsoftmarginloss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/multilabelsoftmarginloss.hpp>
#include <numeric>

struct MultilabelSoftMarginLossTestCase
{
    std::vector<size_t> dims;
};

std::vector<MultilabelSoftMarginLossTestCase> MultilabelSoftMarginLossTestConfigs()
{
    // clang-format off
    return {
    {{22, 12} }, 
    {{75, 19} }, 
    {{33, 4} }, 
    {{54, 7} }, 
    {{87, 23} }, 
    {{10, 3} }, 
    {{341, 11} }, 
    {{564, 17} }, 
    {{289, 2} }, 
    {{456, 8} }, 
    {{711, 15} }, 
    {{987, 22} }, 
    {{1324, 6} }, 
    {{9456, 13} }, 
    {{7532, 20} }, 
    {{8451, 14} }, 
    {{2964, 21} }, 
    {{4987, 1} }, 
    {{15432, 10} }, 
    {{29876, 18} }, 
    {{73915, 5} }, 
    {{58241, 9} }, 
    {{19432, 16} }, 
    {{87009, 7} }, 
    {{123456, 24} }, 
    {{543210, 12} }, 
    {{389124, 19} }, 
    {{678234, 11} }, 
    {{912345, 14} }, 
    {{456789, 8} }, 
    };
    // clang-format on
}

template <typename T = float>
struct MultilabelSoftMarginLossUnreducedForwardTest
    : public ::testing::TestWithParam<MultilabelSoftMarginLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims = config.dims;

        auto gen_in_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input             = tensor<T>{in_dims};
        std::generate(input.begin(), input.end(), gen_in_value);

        auto gen_target_value = [](auto...) { return prng::gen_A_to_B<int32_t>(0, 2); };
        target                = tensor<T>{in_dims};
        std::generate(target.begin(), target.end(), gen_target_value);

        auto gen_weight_value = [](auto...) {
            return prng::gen_descreet_uniform_sign<T>(1e-2, 100);
        };
        // input is (N, C) -> weight is (C)
        weight = tensor<T>{std::vector<size_t>{in_dims[1]}};
        std::generate(weight.begin(), weight.end(), gen_weight_value);

        // input is (N, C) -> output is (N)
        output = tensor<T>{std::vector<size_t>{in_dims[0]}};
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<T>{std::vector<size_t>{in_dims[0]}};
        std::fill(ref_output.begin(), ref_output.end(), 0);

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        weight_dev = handle.Write(weight.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_multilabelsoftmarginloss_unreduced_forward<T>(input, target, weight, ref_output);
        miopenStatus_t status;

        status = miopen::MultilabelSoftMarginLossUnreducedForward(handle,
                                                                  input.desc,
                                                                  input_dev.get(),
                                                                  target.desc,
                                                                  target_dev.get(),
                                                                  weight.desc,
                                                                  weight_dev.get(),
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
    MultilabelSoftMarginLossTestCase config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> weight;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T = float>
struct MultilabelSoftMarginLossReducedForwardTest
    : public ::testing::TestWithParam<MultilabelSoftMarginLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims = config.dims;

        auto N = in_dims[0];
        if(std::is_same<T, half_float::half>::value && N > 20000)
        {
            std::cerr
                << "For fp16 forward reduction test, too many elements in input tensor can"
                   "lead to fp16 overflow or underflow. If reduction mean, divisor is very "
                   "big lead to underflow. If reduction sum, result is too big lead to overflow."
                << std::endl;
            GTEST_SKIP();
        }

        auto gen_in_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        input             = tensor<T>{in_dims};
        std::generate(input.begin(), input.end(), gen_in_value);

        auto gen_target_value = [](auto...) { return prng::gen_A_to_B<int32_t>(0, 2); };
        target                = tensor<T>{in_dims};
        std::generate(target.begin(), target.end(), gen_target_value);

        auto gen_weight_value = [](auto...) {
            return prng::gen_descreet_uniform_sign<T>(1e-2, 100);
        };
        // input is (N, C) -> weight is (C)
        weight = tensor<T>{std::vector<size_t>{in_dims[1]}};
        std::generate(weight.begin(), weight.end(), gen_weight_value);

        // Tensor with 1 element to store result after reduce
        output = tensor<T>{std::vector<size_t>{1}};
        std::fill(output.begin(), output.end(), 0);

        ref_output = tensor<T>{std::vector<size_t>{1}};
        std::fill(ref_output.begin(), ref_output.end(), 0);

        // Mean reduction. To test with sum reduction, change divisor to 1
        divisor = input.desc.GetLengths()[0];

        ws_sizeInBytes = miopen::GetMultilabelSoftMarginLossForwardWorkspaceSize(
            handle,
            input.desc,
            target.desc,
            weight.desc,
            output.desc,
            divisor == 1 ? MIOPEN_LOSS_REDUCTION_SUM : MIOPEN_LOSS_REDUCTION_MEAN);
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
        weight_dev    = handle.Write(weight.data);
        output_dev    = handle.Write(output.data);
        workspace_dev = handle.Write(workspace.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_multilabelsoftmarginloss_reduced_forward<T>(
            input, target, weight, ref_output, ref_workspace, divisor);

        miopenStatus_t status;
        status = miopen::MultilabelSoftMarginLossForward(handle,
                                                         workspace_dev.get(),
                                                         ws_sizeInBytes,
                                                         input.desc,
                                                         input_dev.get(),
                                                         target.desc,
                                                         target_dev.get(),
                                                         weight.desc,
                                                         weight_dev.get(),
                                                         output.desc,
                                                         output_dev.get(),
                                                         divisor == 1 ? MIOPEN_LOSS_REDUCTION_SUM
                                                                      : MIOPEN_LOSS_REDUCTION_MEAN);
        EXPECT_EQ(status, miopenStatusSuccess);

        // Write from GPU to CPU
        workspace.data = handle.Read<T>(workspace_dev, workspace.data.size());
        output.data    = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);
        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance "
                                               "Error:"
                                            << error << ",  Threshold x 10: " << threshold * 10;
    }
    MultilabelSoftMarginLossTestCase config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> weight;
    tensor<T> output;
    tensor<T> workspace;

    tensor<T> ref_output;
    tensor<T> ref_workspace;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    float divisor;
    size_t ws_sizeInBytes;
};
