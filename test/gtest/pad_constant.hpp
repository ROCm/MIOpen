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
// // #include <iostream>
// #include <cstdint>
// #include <miopen/miopen.h>
// #include <miopen/pad_constant.hpp>

// #include <gtest/gtest.h>

// #include "cpu_pad_constant.hpp"
// #include "get_handle.hpp"
// #include "random.hpp"
// #include "tensor_holder.hpp"
// #include "verify.hpp"

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/pad_constant.hpp>

#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include "cpu_pad_constant.hpp"

#define MAX_POS_PADDING 6
#define MIN_NEG_PADDING 6

struct PadConstantTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;

    bool is_contiguous    = false;
    uint64_t padding_size = 2;
    double padding_value  = 0;

    friend std::ostream& operator<<(std::ostream& os, const PadConstantTestCase& tc)
    {
        return os << "(N: " << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << ")";
    }

    std::vector<size_t> GetInput()
    {
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, W});
        }
        else if(N != 0)
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }

    uint64_t GetPaddingSize() const { return padding_size; }
    double GetPaddingValue() const { return padding_value; }

    std::vector<size_t> ComputeStrides(std::vector<size_t> input_dims) const
    {
        if(!is_contiguous)
            std::swap(input_dims.front(), input_dims.back());
        std::vector<size_t> strides(input_dims.size());
        strides.back() = 1;
        for(int i = input_dims.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * input_dims[i + 1];
        if(!is_contiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

std::vector<PadConstantTestCase> PadConstantTestConfigs()
{
    return {
        // 2D
        {4, 0, 0, 0, 4},
        {4, 0, 0, 0, 4, false, 2, 0.12},
        {8, 0, 0, 0, 8, false, 2, 1},

        // 3D
        {8, 512, 0, 0, 384},
        {8, 511, 0, 0, 1},
        {16, 512, 0, 0, 8, false, 4, 2.3},

        // 4D
        {8, 16, 0, 32, 32},
        {8, 16, 0, 32, 32, false, 4, 0.18127938797213},

        // 5D
        {8, 4, 16, 32, 32, false, 8, 2},
        {10, 4, 16, 32, 32},
    };
}

template <typename T>
struct PadConstantFwdTest : public ::testing::TestWithParam<PadConstantTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle     = get_handle();
        config            = GetParam();
        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        // Generate input
        auto input_dims                   = config.GetInput();
        std::vector<size_t> input_strides = config.ComputeStrides(input_dims);
        input = tensor<T>{input_dims, input_strides}.generate(in_gen_value);

        uint64_t padding_length = config.GetPaddingSize();
        int64_t min_in_dim      = *std::min_element(input_dims.begin(), input_dims.end());
        int64_t min_padding =
            -std::min((int64_t)MIN_NEG_PADDING, std::min((int64_t)0, (min_in_dim / 2 - 1)));
        for(uint64_t i = 0; i < padding_length; i++)
        {
            padding.push_back(prng::gen_A_to_B<int64_t>(min_padding, MAX_POS_PADDING));
        }

        // Generate output
        std::vector<size_t> output_dims = input_dims;
        for(uint64_t i = 0; i < padding_length / 2; i++)
        {
            uint64_t idx = input_dims.size() - i - 1;
            output_dims[idx] += padding[i * 2] + padding[i * 2 + 1];
        }

        output = tensor<T>{output_dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{output_dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        // Run cpu
        cpu_pad_constant_fwd(input, ref_output, padding, static_cast<T>(config.GetPaddingValue()));

        // Run kernel
        status = miopen::PadConstantForward(handle,
                                            input.desc,
                                            output.desc,
                                            input_dev.get(),
                                            output_dev.get(),
                                            padding.data(),
                                            padding.size(),
                                            // padding_value
                                            config.GetPaddingValue());

        ASSERT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        // Verify output_tensor
        double threshold = GetTolerance();
        auto error       = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold) << "Error output beyond tolerance Error: " << error
                                    << ", Threshold: " << threshold << std::endl;
    }

    PadConstantTestCase config;
    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    std::vector<int64_t> padding;
};

template <typename T>
struct PadConstantBwdTest : public ::testing::TestWithParam<PadConstantTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        auto input_grad_dims = config.GetInput();

        // Generate padding
        auto padding_length = config.GetPaddingSize();
        int64_t min_in_dim  = *std::min_element(input_grad_dims.begin(), input_grad_dims.end());
        int64_t min_padding =
            -std::min((int64_t)MIN_NEG_PADDING, std::min((int64_t)0, (min_in_dim / 2 - 1)));
        for(auto i = 0; i < padding_length; i++)
        {
            padding.push_back(prng::gen_A_to_B<int64_t>(min_padding, MAX_POS_PADDING));
        }

        // Generate output
        std::vector<size_t> output_grad_dims = input_grad_dims;

        for(uint64_t i = 0; i < padding_length / 2; i++)
        {
            int idx = input_grad_dims.size() - i - 1;
            output_grad_dims[idx] += padding[i * 2] + padding[i * 2 + 1];
        }

        std::vector<size_t> output_grad_strides = config.ComputeStrides(output_grad_dims);

        output_grad = tensor<T>{output_grad_dims, output_grad_strides};
        std::fill(output_grad.begin(), output_grad.end(), static_cast<T>(1.0f));

        input_grad = tensor<T>{input_grad_dims};

        ref_input_grad = tensor<T>{input_grad_dims};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        input_grad_dev  = handle.Write(input_grad.data);
        output_grad_dev = handle.Write(output_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        // Run cpu
        cpu_pad_constant_bwd(output_grad, ref_input_grad, padding);

        // Run kernel
        status = miopen::PadConstantBackward(handle,
                                             input_grad.desc,
                                             output_grad.desc,
                                             input_grad_dev.get(),
                                             output_grad_dev.get(),
                                             padding.data(),
                                             padding.size());

        ASSERT_EQ(status, miopenStatusSuccess);

        // Copy output data from device to host
        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    double GetTolerance()
    {
        double tolerance = std::numeric_limits<T>::epsilon() * 10;
        return tolerance;
    }

    void Verify()
    {
        // Verify output_tensor
        double threshold = GetTolerance();
        auto error       = miopen::rms_range(ref_input_grad, input_grad);

        ASSERT_EQ(miopen::range_distance(ref_input_grad), miopen::range_distance(input_grad));
        EXPECT_LT(error, threshold) << "Error output beyond tolerance Error: " << error
                                    << ", Threshold: " << threshold << std::endl;
    }

    PadConstantTestCase config;
    tensor<T> input_grad;
    tensor<T> output_grad;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;

    std::vector<int64_t> padding;
};
