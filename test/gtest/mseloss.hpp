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

#include "cpu_mseloss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/allocator.hpp>
#include <miopen/mseloss.hpp>

#include <cstddef>
#include <limits>
#include <vector>

struct MSELossTestCase
{
    std::vector<size_t> lengths;
    miopenLossReductionMode_t reduction;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const MSELossTestCase& tc)
    {
        os << " lengths:";
        for(int i = 0; i < tc.lengths.size(); i++)
        {
            auto input = tc.lengths[i];
            if(i != 0)
                os << ",";
            os << input;
        }
        os << " reduction:" << tc.reduction << " contiguous:" << tc.isContiguous;
        return os;
    }
};

inline std::vector<MSELossTestCase> MSELossTestFwdConfigs()
{
    return {
        {{10000}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{10000}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{1000000}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{25, 100}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{25, 100}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{2000, 3000}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{1, 2, 3}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{1, 2, 3}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{8, 8, 8}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{8, 8, 8}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{16, 128, 384}, MIOPEN_LOSS_REDUCTION_MEAN, true},
        {{25, 100, 100}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{1, 2, 3, 4}, MIOPEN_LOSS_REDUCTION_SUM, true},
        {{1, 2, 3, 4}, MIOPEN_LOSS_REDUCTION_MEAN, true},
        {{8, 8, 8, 8}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{8, 8, 8, 8}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{16, 32, 32, 32}, MIOPEN_LOSS_REDUCTION_MEAN, true},
        {{1, 1, 16, 1024}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{1, 1, 16, 1024}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{16, 16, 32, 32, 2}, MIOPEN_LOSS_REDUCTION_MEAN, true},
        {{16, 16, 32, 32, 256}, MIOPEN_LOSS_REDUCTION_MEAN, false},
    };
}

inline std::vector<MSELossTestCase> MSELossTestBwdConfigs()
{
    return {
        {{10000, 2}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{10000, 2}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{10000, 2}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{2, 1000000}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{2, 1000000}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{25, 100}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{25, 100}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{25, 100}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{2000, 3000}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{2000, 3000}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{2, 3}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{2, 3}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{2, 3}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{8, 8}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{8, 8}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{8, 8}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{128, 384}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{128, 384}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{128, 384}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{100, 100}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{100, 100}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{100, 100}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{3, 4}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{3, 4}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{3, 4}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{2, 8}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{2, 8}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{2, 8}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{32, 32}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{32, 32}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{32, 32}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{16, 1024}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{16, 1024}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{16, 1024}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{32, 2}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{32, 2}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{32, 2}, MIOPEN_LOSS_REDUCTION_MEAN, false},
        {{32, 256}, MIOPEN_LOSS_REDUCTION_NONE, false},
        {{32, 256}, MIOPEN_LOSS_REDUCTION_SUM, false},
        {{32, 256}, MIOPEN_LOSS_REDUCTION_MEAN, false},
    };
}

inline std::vector<size_t> GetStrides(std::vector<size_t> input, bool contiguous)
{
    if(!contiguous)
        std::swap(input.front(), input.back());
    std::vector<size_t> strides(input.size());
    strides.back() = 1;
    for(int i = input.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * input[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <class T>
struct MSELossTestForward : public ::testing::TestWithParam<MSELossTestCase>
{
protected:
    MSELossTestCase mseloss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<T> output_ref;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    miopenLossReductionMode_t reduction;

    size_t ws_sizeInBytes;

    void SetUp() override
    {
        auto&& handle   = get_handle();
        mseloss_config  = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 101); };

        auto in_dims = mseloss_config.lengths;
        auto strides = GetStrides(in_dims, mseloss_config.isContiguous);

        input  = tensor<T>{in_dims, strides}.generate(gen_value1);
        target = tensor<T>{in_dims, strides}.generate(gen_value2);

        reduction = mseloss_config.reduction;

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            output     = tensor<T>{in_dims};
            output_ref = tensor<T>{in_dims};
        }
        else
        {
            output     = tensor<T>{{1}};
            output_ref = tensor<T>{{1}};
        }

        ws_sizeInBytes =
            miopen::GetMSELossForwardWorkspaceSize(handle, input.desc, output.desc, reduction);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();
        workspace_dev = handle.Create(ws_sizeInBytes);

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Create(output.desc.GetNumBytes());
    }

    void RunTest()
    {
        cpu_mseloss_forward<T, 5>(input, target, output_ref, reduction);

        auto&& handle = get_handle();
        auto status   = miopen::MSELossForward(handle,
                                             workspace_dev.get(),
                                             ws_sizeInBytes,
                                             input.desc,
                                             input_dev.get(),
                                             target.desc,
                                             target_dev.get(),
                                             output.desc,
                                             output_dev.get(),
                                             reduction);

        ASSERT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        auto error = miopen::rms_range(output_ref, output);
        EXPECT_EQ(miopen::range_distance(output_ref), miopen::range_distance(output));
        EXPECT_LT(error, std::numeric_limits<T>::epsilon())
            << "Forward outputs do not match each other. Error:" << error;
    }
};

template <class T>
struct MSELossTestBackward : public ::testing::TestWithParam<MSELossTestCase>
{
protected:
    MSELossTestCase mseloss_config;

    tensor<T> input;
    tensor<T> target;

    tensor<T> output_grad;
    tensor<T> input_grad;
    tensor<T> target_grad;
    tensor<T> input_grad_ref;
    tensor<T> target_grad_ref;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr target_grad_dev;

    miopen::Allocator::ManageDataPtr workspace_dev;

    miopenLossReductionMode_t reduction;

    void SetUp() override
    {
        auto&& handle   = get_handle();
        mseloss_config  = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 101); };

        auto in_dims = mseloss_config.lengths;
        auto strides = GetStrides(in_dims, mseloss_config.isContiguous);

        input  = tensor<T>{in_dims, strides}.generate(gen_value1);
        target = tensor<T>{in_dims, strides}.generate(gen_value2);

        input_grad      = tensor<T>{in_dims};
        target_grad     = tensor<T>{in_dims};
        input_grad_ref  = tensor<T>{in_dims};
        target_grad_ref = tensor<T>{in_dims};

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);

        reduction = mseloss_config.reduction;

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            output_grad = tensor<T>{in_dims};
        }
        else
        {
            output_grad = tensor<T>{{1}};
        }

        std::fill(output_grad.begin(), output_grad.end(), static_cast<T>(1.0f));

        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(target_grad.begin(), target_grad.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(
            input_grad_ref.begin(), input_grad_ref.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(
            target_grad_ref.begin(), target_grad_ref.end(), std::numeric_limits<T>::quiet_NaN());

        output_grad_dev = handle.Write(output_grad.data);
        input_grad_dev  = handle.Write(input_grad.data);
        target_grad_dev = handle.Write(target_grad.data);
    }

    void RunTest()
    {
        cpu_mseloss_backward<T, 5>(
            input, target, output_grad, input_grad_ref, target_grad_ref, reduction);

        auto&& handle = get_handle();
        auto status   = miopen::MSELossBackward(handle,
                                              input.desc,
                                              input_dev.get(),
                                              target.desc,
                                              target_dev.get(),
                                              output_grad.desc,
                                              output_grad_dev.get(),
                                              input_grad.desc,
                                              input_grad_dev.get(),
                                              target_grad.desc,
                                              target_grad_dev.get(),
                                              reduction);
        ASSERT_EQ(status, miopenStatusSuccess);

        input_grad.data  = handle.Read<T>(input_grad_dev, input_grad.data.size());
        target_grad.data = handle.Read<T>(target_grad_dev, target_grad.data.size());
    }

    void Verify()
    {
        auto error_input_grad = miopen::rms_range(input_grad_ref, input_grad);
        EXPECT_EQ(miopen::range_distance(input_grad_ref), miopen::range_distance(input_grad));
        EXPECT_LT(error_input_grad, std::numeric_limits<T>::epsilon())
            << "Backward input gradients do not match each other. Error:" << error_input_grad;

        auto error_target_grad = miopen::rms_range(target_grad_ref, target_grad);
        EXPECT_EQ(miopen::range_distance(target_grad_ref), miopen::range_distance(target_grad));
        EXPECT_LT(error_target_grad, std::numeric_limits<T>::epsilon())
            << "Backward target gradients do not match each other. Error:" << error_target_grad;
    }
};
