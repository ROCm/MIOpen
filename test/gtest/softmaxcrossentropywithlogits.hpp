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
#include "cpu_softmaxcrossentropywithlogits.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/softmaxcrossentropywithlogits.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(size_t i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

struct SoftmaxCrossEntropyWithLogitsTestCase
{
    std::vector<size_t> input;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os,
                                    const SoftmaxCrossEntropyWithLogitsTestCase& tc)
    {
        return os << " input:" << tc.input << " contiguous:" << tc.contiguous;
    }

    std::vector<size_t> GetInput() const { return input; }
};

inline std::vector<SoftmaxCrossEntropyWithLogitsTestCase> SoftmaxCrossEntropyWithLogitsTestConfigs()
{
    return {
        {{2000, 3000}, true},
        {{768, 200}, true},
        {{768, 128}, true},
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

// FORWARD TEST
template <typename T = float>
struct SoftmaxCrossEntropyWithLogitsTestFwd
    : public ::testing::TestWithParam<SoftmaxCrossEntropyWithLogitsTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                        = get_handle();
        softmaxcrossentropywithlogits_config = GetParam();

        auto in_dim       = softmaxcrossentropywithlogits_config.GetInput();
        auto tar_dim      = in_dim;
        auto out_dim      = std::vector<size_t>({in_dim[0]});
        auto backprop_dim = in_dim;
        auto contiguous   = softmaxcrossentropywithlogits_config.contiguous;
        auto num_classes  = in_dim[1];

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-5.0f), static_cast<T>(1.0f));
        };

        auto in_strides = GetStrides(in_dim, contiguous);
        input           = tensor<T>{in_dim, in_strides}.generate(gen_input_value);

        auto tar_strides = GetStrides(tar_dim, true);
        target           = tensor<T>{tar_dim, tar_strides};
        for(size_t i = 0; i < tar_dim[0]; i++)
        {
            for(size_t j = 0; j < tar_dim[1]; j++)
            {
                if(j == i % num_classes)
                    target[i * num_classes + j] = (static_cast<T>(1.0f));
                else
                    target[i * num_classes + j] = (static_cast<T>(0.0f));
            }
        }

        auto out_strides = GetStrides(out_dim, true);
        output           = tensor<T>{out_dim, out_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{out_dim, out_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        auto backprop_strides = GetStrides(backprop_dim, true);
        backprop              = tensor<T>{backprop_dim, backprop_strides};
        std::fill(backprop.begin(), backprop.end(), std::numeric_limits<T>::quiet_NaN());

        ref_backprop = tensor<T>{backprop_dim, backprop_strides};
        std::fill(ref_backprop.begin(), ref_backprop.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev    = handle.Write(input.data);
        target_dev   = handle.Write(target.data);
        output_dev   = handle.Write(output.data);
        backprop_dev = handle.Write(backprop.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_softmaxcrossentropywithlogits_forward<T>(input, target, ref_output, ref_backprop);

        status = miopen::softmaxcrossentropywithlogits::SoftmaxCrossEntropyWithLogitsForward(
            handle,
            input.desc,
            input_dev.get(),
            target.desc,
            target_dev.get(),
            output.desc,
            output_dev.get(),
            backprop.desc,
            backprop_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data   = handle.Read<T>(output_dev, output.data.size());
        backprop.data = handle.Read<T>(backprop_dev, backprop.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error          = miopen::rms_range(ref_output, output);
        auto backprop_error = miopen::rms_range(ref_backprop, backprop);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold * 10) << "Error output beyond tolerance Error:" << error
                                         << ",  Thresholdx10: " << threshold * 10;
        ASSERT_EQ(miopen::range_distance(ref_backprop), miopen::range_distance(backprop));
        EXPECT_LT(backprop_error, threshold * 10)
            << "Error backprop beyond tolerance Error:" << backprop_error
            << ",  Thresholdx10: " << threshold * 10;
    }
    SoftmaxCrossEntropyWithLogitsTestCase softmaxcrossentropywithlogits_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<T> ref_output;
    tensor<T> backprop;
    tensor<T> ref_backprop;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr backprop_dev;
};

// BACKWARD TEST
template <typename T = float>
struct SoftmaxCrossEntropyWithLogitsTestBwd
    : public ::testing::TestWithParam<SoftmaxCrossEntropyWithLogitsTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle                        = get_handle();
        softmaxcrossentropywithlogits_config = GetParam();

        auto in_dim       = softmaxcrossentropywithlogits_config.GetInput();
        auto backprop_dim = in_dim;
        auto out_grad_dim = std::vector<size_t>({in_dim[0]});
        auto in_grad_dim  = in_dim;
        auto tar_grad_dim = in_dim;

        auto gen_output_grad_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-5.0f), static_cast<T>(5.0f));
        };

        auto gen_backprop_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-5.0f), static_cast<T>(5.0f));
        };

        auto gen_input_value = [](auto...) {
            return prng::gen_A_to_B<T>(static_cast<T>(-5.0f), static_cast<T>(5.0f));
        };

        auto out_grad_strides = GetStrides(out_grad_dim, true);
        output_grad = tensor<T>{out_grad_dim, out_grad_strides}.generate(gen_output_grad_value);

        auto backprop_strides = GetStrides(backprop_dim, true);
        backprop = tensor<T>{backprop_dim, backprop_strides}.generate(gen_backprop_value);

        auto in_strides = GetStrides(in_dim, true);
        input           = tensor<T>{in_dim, in_strides}.generate(gen_input_value);

        input_grad = tensor<T>{in_grad_dim, in_strides};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        target_grad = tensor<T>{tar_grad_dim, in_strides};
        std::fill(target_grad.begin(), target_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{in_grad_dim, in_strides};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_target_grad = tensor<T>{tar_grad_dim, in_strides};
        std::fill(
            ref_target_grad.begin(), ref_target_grad.end(), std::numeric_limits<T>::quiet_NaN());

        output_grad_dev = handle.Write(output_grad.data);
        backprop_dev    = handle.Write(backprop.data);
        input_dev       = handle.Write(input.data);
        input_grad_dev  = handle.Write(input_grad.data);
        target_grad_dev = handle.Write(target_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        cpu_softmaxcrossentropywithlogits_backward<T>(
            output_grad, backprop, input, ref_input_grad, ref_target_grad, true, true);

        status = miopen::softmaxcrossentropywithlogits::SoftmaxCrossEntropyWithLogitsBackward(
            handle,
            output_grad.desc,
            output_grad_dev.get(),
            backprop.desc,
            backprop_dev.get(),
            input.desc,
            input_dev.get(),
            input_grad.desc,
            input_grad_dev.get(),
            target_grad.desc,
            target_grad_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        input_grad.data  = handle.Read<T>(input_grad_dev, input_grad.data.size());
        target_grad.data = handle.Read<T>(target_grad_dev, target_grad.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();

        auto error1 = miopen::rms_range(ref_input_grad, input_grad);
        auto error2 = miopen::rms_range(ref_target_grad, target_grad);

        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error1, threshold * 10)
            << "Error input gradient beyond tolerance Error: " << error1
            << ",  Tolerance: " << threshold * 10;
        ASSERT_EQ(miopen::range_distance(ref_target_grad), miopen::range_distance(target_grad));
        EXPECT_LT(error2, threshold * 10)
            << "Error target gradient beyond tolerance Error: " << error2
            << ",  Tolerance: " << threshold * 10;
    }
    SoftmaxCrossEntropyWithLogitsTestCase softmaxcrossentropywithlogits_config;

    tensor<T> output_grad;
    tensor<T> backprop;
    tensor<T> input;
    tensor<T> input_grad;
    tensor<T> target_grad;

    tensor<T> ref_input_grad;
    tensor<T> ref_target_grad;

    miopen::Allocator::ManageDataPtr output_grad_dev;
    miopen::Allocator::ManageDataPtr backprop_dev;
    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr target_grad_dev;
};
