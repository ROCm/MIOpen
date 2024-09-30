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

#include "get_handle.hpp"
#include "miopen/miopen.h"
#include "cpu_mseloss.hpp"
#include "random.hpp"
#include "miopen/allocator.hpp"
#include "tensor_holder.hpp"
#include "miopen/mseloss.hpp"
#include "verify.hpp"
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

struct MSELossTestCase
{
    std::vector<size_t> lengths;
    float divisor;
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
        os << " divisor:" << tc.divisor << " contiguous:" << tc.isContiguous;
        return os;
    }
};

std::vector<MSELossTestCase> MSELossTestConfigs()
{
    // clang-format off
    return {
            {{10000}, 10000.0f, false},
            {{1000000}, 1.0f, false},
            {{25, 100}, 25000.0f, false},
            {{2000,3000}, 1.0f, false},
            {{1, 2,3}, 1.0f, false},
            {{8, 8,8}, 1.0f, false},
            {{16, 128,384}, 1.0f, true},
            {{25,100,100}, 1.0f, false},
            {{1,2,3,4}, 1.0f, true},
            {{8, 8, 8, 8}, 1.0f, false},
            {{16, 32, 32, 32}, 1.0f,true},
            {{1,1,16,1024}, 1.0f, false},
            {{16, 16, 32, 32, 2}, 1.0f, true},
            {{16, 16, 32, 32, 256}, 1.0f, false}
            };
    // clang-format on
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
struct MSELossTest : public ::testing::TestWithParam<MSELossTestCase>
{
protected:
    MSELossTestCase mseloss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<T> output_ref;

    tensor<T> input_grad;
    tensor<T> target_grad;
    tensor<T> input_grad_ref;
    tensor<T> target_grad_ref;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr target_grad_dev;

    float divisor;

    void SetUp() override
    {
        auto&& handle  = get_handle();
        mseloss_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto in_dims = mseloss_config.lengths;
        auto strides = GetStrides(in_dims, mseloss_config.isContiguous);

        input  = tensor<T>{in_dims, strides}.generate(gen_value);
        target = tensor<T>{in_dims, strides}.generate(gen_value);

        input_grad      = tensor<T>{in_dims};
        target_grad     = tensor<T>{in_dims};
        input_grad_ref  = tensor<T>{in_dims};
        target_grad_ref = tensor<T>{in_dims};

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);

        divisor = mseloss_config.divisor;

        if(divisor == 0.0f)
        {
            output     = tensor<T>{in_dims};
            output_ref = tensor<T>{in_dims};
        }
        else
        {
            output     = tensor<T>{{1}};
            output_ref = tensor<T>{{1}};
        }
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(output_ref.begin(), output_ref.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(target_grad.begin(), target_grad.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(
            input_grad_ref.begin(), input_grad_ref.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(
            target_grad_ref.begin(), target_grad_ref.end(), std::numeric_limits<T>::quiet_NaN());

        output_dev      = handle.Write(output.data);
        input_grad_dev  = handle.Write(input_grad.data);
        target_grad_dev = handle.Write(target_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        auto outDesc  = output.desc;

        // forward portion
        size_t workspace_in_bytes = 0;
        auto status               = miopenGetMSELossForwardWorkspaceSize(
            &handle, &input.desc, &target.desc, &workspace_in_bytes);

        if(status != miopenStatusSuccess)
        {
            std::cout << "Error: failed to obtain workspace size" << std::endl;
        }

        workspace_dev = handle.Create(workspace_in_bytes);

        cpu_mseloss<T>(input.desc,
                       target.desc,
                       output.desc,
                       input.data.data(),
                       target.data.data(),
                       output_ref.data.data(),
                       divisor);

        status = MSELossForward(handle,
                                input.desc,
                                target.desc,
                                output.desc,
                                input_dev.get(),
                                target_dev.get(),
                                output_dev.get(),
                                workspace_dev.get(),
                                divisor);

        ASSERT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());

        if(input.desc.GetLengths().size() == 2) // only case we support
        {
            cpu_mseloss_backward<T>(input.desc,
                                    target.desc,
                                    output.desc,
                                    input_grad.desc,
                                    target_grad.desc,
                                    input.data.data(),
                                    target.data.data(),
                                    output_ref.data.data(),
                                    input_grad_ref.data.data(),
                                    target_grad_ref.data.data(),
                                    divisor);

            status = MSELossBackward(handle,
                                     input.desc,
                                     target.desc,
                                     output.desc,
                                     input_grad.desc,
                                     target_grad.desc,
                                     input_dev.get(),
                                     target_dev.get(),
                                     output_dev.get(),
                                     input_grad_dev.get(),
                                     target_grad_dev.get(),
                                     divisor);
            ASSERT_EQ(status, miopenStatusSuccess);
        }

        if(input.desc.GetLengths().size() == 2)
        {
            input_grad.data  = handle.Read<T>(input_grad_dev, input_grad.data.size());
            target_grad.data = handle.Read<T>(target_grad_dev, target_grad.data.size());
        }
    }

    void Verify()
    {
        // Forward verification
        auto error = miopen::rms_range(output_ref, output);
        EXPECT_EQ(miopen::range_distance(output_ref), miopen::range_distance(output));
        EXPECT_LT(error, std::numeric_limits<T>::epsilon())
            << "Forward outputs do not match each other. Error:" << error;

        if(input.desc.GetLengths().size() == 2) // only case we support backward pass
        {
            error = miopen::rms_range(input_grad_ref, input_grad);
            EXPECT_EQ(miopen::range_distance(input_grad_ref), miopen::range_distance(input_grad));
            EXPECT_LT(error, std::numeric_limits<T>::epsilon())
                << "Backward input gradients do not match each other. Error:" << error;

            error = miopen::rms_range(target_grad_ref, target_grad);
            EXPECT_EQ(miopen::range_distance(target_grad_ref), miopen::range_distance(target_grad));
            EXPECT_LT(error, std::numeric_limits<T>::epsilon())
                << "Backward target gradients do not match each other. Error:" << error;
        }
    }
};
