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
    miopenLossReductionMode_t reduction_mode;
};

std::vector<MultilabelSoftMarginLossTestCase> MultilabelSoftMarginLossTestConfigs()
{
    // clang-format off
    return {
    {{22, 12}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{75, 19}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{33, 4}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{54, 7}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{87, 23}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{10, 3}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{341, 11}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{564, 17}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{289, 2}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{456, 8}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{711, 15}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{987, 22}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{1324, 6}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{9456, 13}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{7532, 20}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{8451, 14}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{2964, 21}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{4987, 1}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{15432, 10}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{29876, 18}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{73915, 5}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{58241, 9}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{19432, 16}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{87009, 7}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{123456, 24}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{543210, 12}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{389124, 19}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{678234, 11}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{912345, 14}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{456789, 8}, MIOPEN_LOSS_REDUCTION_MEAN }, 
    {{22, 12}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{75, 19}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{33, 4}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{54, 7}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{87, 23}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{10, 3}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{341, 11}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{564, 17}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{289, 2}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{456, 8}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{711, 15}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{987, 22}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{1324, 6}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{9456, 13}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{7532, 20}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{8451, 14}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{2964, 21}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{4987, 1}, MIOPEN_LOSS_REDUCTION_SUM }, 
    {{22, 12}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{75, 19}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{33, 4}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{54, 7}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{87, 23}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{10, 3}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{341, 11}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{564, 17}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{289, 2}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{456, 8}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{711, 15}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{987, 22}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{1324, 6}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{9456, 13}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{7532, 20}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{8451, 14}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{2964, 21}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{4987, 1}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{15432, 10}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{29876, 18}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{73915, 5}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{58241, 9}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{19432, 16}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{87009, 7}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{123456, 24}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{543210, 12}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{389124, 19}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{678234, 11}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{912345, 14}, MIOPEN_LOSS_REDUCTION_NONE }, 
    {{456789, 8}, MIOPEN_LOSS_REDUCTION_NONE }, 
    };
    // clang-format on
}

template <typename T = float>
struct MultilabelSoftMarginLossForwardTest
    : public ::testing::TestWithParam<MultilabelSoftMarginLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto in_dims   = config.dims;
        reduction_mode = config.reduction_mode;

        auto gen_in_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(0.1, 50); };
        input             = tensor<T>{in_dims};
        std::generate(input.begin(), input.end(), gen_in_value);
        input_dev = handle.Write(input.data);

        auto gen_target_value = [](auto...) { return prng::gen_A_to_B<int32_t>(0, 2); };
        target                = tensor<T>{in_dims};
        std::generate(target.begin(), target.end(), gen_target_value);
        target_dev = handle.Write(target.data);

        auto gen_weight_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(0.1, 50); };
        // input is (N, C) -> weight is (C)
        weight = tensor<T>{std::vector<size_t>{in_dims[1]}};
        std::generate(weight.begin(), weight.end(), gen_weight_value);
        weight_dev = handle.Write(weight.data);

        if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
        {
            // input is (N, C) -> output is (N)
            output     = tensor<T>{std::vector<size_t>{in_dims[0]}};
            ref_output = tensor<T>{std::vector<size_t>{in_dims[0]}};
        }
        else
        {
            // Tensor with 1 element to store result after reduce
            output     = tensor<T>{std::vector<size_t>{1}};
            ref_output = tensor<T>{std::vector<size_t>{1}};
        }
        std::fill(output.begin(), output.end(), 0);
        std::fill(ref_output.begin(), ref_output.end(), 0);
        output_dev = handle.Write(output.data);

        if(reduction_mode != MIOPEN_LOSS_REDUCTION_NONE)
        {
            ws_sizeInBytes = miopen::GetMultilabelSoftMarginLossForwardWorkspaceSize(
                handle, input.desc, target.desc, weight.desc, output.desc, reduction_mode);
            if(ws_sizeInBytes == static_cast<size_t>(-1))
                GTEST_SKIP();
            workspace = tensor<T>{std::vector<size_t>{ws_sizeInBytes / sizeof(T)}};
            std::fill(workspace.begin(), workspace.end(), 0);
            workspace_dev = handle.Write(workspace.data);
        }
    }
    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;
        if(reduction_mode == MIOPEN_LOSS_REDUCTION_NONE)
        {
            cpu_multilabelsoftmarginloss_unreduced_forward<T>(input, target, weight, ref_output);

            status = miopen::MultilabelSoftMarginLossUnreducedForward(handle,
                                                                      input.desc,
                                                                      input_dev.get(),
                                                                      target.desc,
                                                                      target_dev.get(),
                                                                      weight.desc,
                                                                      weight_dev.get(),
                                                                      output.desc,
                                                                      output_dev.get());
        }
        else
        {
            cpu_multilabelsoftmarginloss_reduced_forward<T>(
                input,
                target,
                weight,
                ref_output,
                (reduction_mode == MIOPEN_LOSS_REDUCTION_MEAN) ? input.desc.GetLengths()[0] : 1);

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
                                                             reduction_mode);
        }
        EXPECT_EQ(status, miopenStatusSuccess);

        // Write from GPU to CPU
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        auto threshold = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;

        auto error = miopen::rms_range(ref_output, output);
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

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    miopenLossReductionMode_t reduction_mode;
    size_t ws_sizeInBytes;
};
