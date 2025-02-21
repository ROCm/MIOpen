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

#pragma once

#include "cpu_logsumexp.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/logsumexp.hpp>

struct LogSumExpTestCase
{
    std::vector<size_t> input_size;
    std::vector<int> reduce_dims;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const LogSumExpTestCase& tc)
    {
        os << "input_size: ";
        for(auto d : tc.input_size)
            os << d << " ";
        os << "reduce_dims: ";
        for(auto d : tc.reduce_dims)
            os << d << " ";
        os << "is_contiguous: " << tc.isContiguous;
        return os;
    }
};

inline std::vector<size_t> ComputeStrides(std::vector<size_t> dims, const bool isContiguous)
{
    if(!isContiguous)
        std::swap(dims.front(), dims.back());
    std::vector<size_t> strides(dims.size());
    strides.back() = 1;
    for(int i = dims.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * dims[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

inline std::vector<LogSumExpTestCase> LogSumExpTestConfigs()
{
    return {
        {{400}, {0}, true},
        {{800}, {0}, true},
        {{12, 40}, {0}, true},
        {{16, 120}, {0}, true},
        {{256, 32}, {0}, true},
        {{1000, 48}, {0}, true},
        {{32, 24}, {0, 1}, true},
        {{12, 36}, {0, 1}, true},
        {{12, 18, 10}, {0}, true},
        {{256, 32, 5}, {0}, true},
        {{32, 80, 5}, {1}, true},
        {{32, 96, 16}, {2}, true},
        {{16, 32, 12}, {0, 1}, true},
        {{36, 256, 6}, {0, 2}, true},
        {{12, 24, 2}, {0, 1, 2}, true},
        {{6, 6, 6, 6}, {0}, true},
        {{16, 32, 16, 32}, {3}, true},
        {{32, 64, 16, 16}, {0, 3}, true},
        {{128, 128, 4, 8}, {0, 3}, true},
        {{256, 256, 2, 4}, {0, 3}, true},
        {{512, 512, 1, 2}, {0, 3}, true},
        {{124, 1024, 1, 1}, {0, 3}, true},
        {{12, 6, 10, 5}, {0, 1, 3}, true},
        {{6, 6, 6, 6, 6}, {0}, true},
        {{12, 12, 12, 12, 12}, {3}, true},
        {{16, 16, 8, 32, 8}, {4}, true},
        {{16, 16, 16, 8, 16}, {0, 1}, true},
        {{12, 16, 2, 6, 6}, {0, 1, 2}, true},
        {{16, 256, 6, 2, 12}, {0, 3, 4}, true},
        {{2, 3, 2, 10, 5}, {0, 1, 2}, true},
        {{8, 16, 24, 10, 5}, {0, 1, 4}, true},
        {{6, 3, 3, 5, 16}, {0, 1, 2, 3}, true},
        {{12, 6, 3, 2, 6}, {0, 1, 2, 3}, true},
        {{16, 8, 16, 2, 2}, {0, 1, 3, 4}, true},
        {{11, 16, 2, 2, 5}, {0, 1, 2, 3}, true},
        {{14, 6, 2, 10, 5}, {0, 1, 2, 4}, true},
        {{16, 2, 2, 2, 4}, {0, 1, 2, 3, 4}, true},
        {{6, 6, 4, 2, 2}, {0, 1, 2, 3, 4}, true},
        {{12, 2, 4, 2, 2}, {0, 1, 2, 3, 4}, true},
        {{2, 3, 12, 1, 5}, {0, 1, 2, 3, 4}, true},
        {{2, 2, 6, 12, 2}, {0, 1, 2, 3, 4}, true},

        {{400}, {0}, false},
        {{800}, {0}, false},
        {{12, 40}, {0}, false},
        {{16, 120}, {0}, false},
        {{256, 32}, {0}, false},
        {{1000, 48}, {0}, false},
        {{32, 24}, {0, 1}, false},
        {{12, 36}, {0, 1}, false},
        {{12, 18, 10}, {0}, false},
        {{256, 32, 5}, {0}, false},
        {{32, 80, 5}, {1}, false},
        {{32, 96, 16}, {2}, false},
        {{16, 32, 12}, {0, 1}, false},
        {{36, 256, 6}, {0, 2}, false},
        {{12, 24, 2}, {0, 1, 2}, false},
        {{6, 6, 6, 6}, {0}, false},
        {{16, 32, 16, 32}, {3}, false},
        {{32, 64, 16, 16}, {0, 3}, false},
        {{128, 128, 4, 8}, {0, 3}, false},
        {{256, 256, 2, 4}, {0, 3}, false},
        {{512, 512, 1, 2}, {0, 3}, false},
        {{124, 1024, 1, 1}, {0, 3}, false},
        {{12, 6, 10, 5}, {0, 1, 3}, false},
        {{6, 6, 6, 6, 6}, {0}, false},
        {{12, 12, 12, 12, 12}, {3}, false},
        {{16, 16, 8, 32, 8}, {4}, false},
        {{16, 16, 16, 8, 16}, {0, 1}, false},
        {{12, 16, 2, 6, 6}, {0, 1, 2}, false},
        {{16, 256, 6, 2, 12}, {0, 3, 4}, false},
        {{2, 3, 2, 10, 5}, {0, 1, 2}, false},
        {{8, 16, 24, 10, 5}, {0, 1, 4}, false},
        {{6, 3, 3, 5, 16}, {0, 1, 2, 3}, false},
        {{12, 6, 3, 2, 6}, {0, 1, 2, 3}, false},
        {{16, 8, 16, 2, 2}, {0, 1, 3, 4}, false},
        {{11, 16, 2, 2, 5}, {0, 1, 2, 3}, false},
        {{14, 6, 2, 10, 5}, {0, 1, 2, 4}, false},
        {{16, 2, 2, 2, 4}, {0, 1, 2, 3, 4}, false},
        {{6, 6, 4, 2, 2}, {0, 1, 2, 3, 4}, false},
        {{12, 2, 4, 2, 2}, {0, 1, 2, 3, 4}, false},
        {{2, 3, 12, 1, 5}, {0, 1, 2, 3, 4}, false},
        {{2, 2, 6, 12, 2}, {0, 1, 2, 3, 4}, false},
    };
}

template <typename T = float>
struct LogSumExpForwardTest : public ::testing::TestWithParam<LogSumExpTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle    = get_handle();
        logsumexp_config = GetParam();
        auto gen_value   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        dims_vector = logsumexp_config.reduce_dims;

        auto input_dims    = logsumexp_config.input_size;
        auto input_strides = ComputeStrides(input_dims, logsumexp_config.isContiguous);

        std::vector<size_t> output_dims(input_dims);
        for(const auto& dim : dims_vector)
            output_dims[dim] = 1;
        auto output_strides = ComputeStrides(output_dims, logsumexp_config.isContiguous);

        input = tensor<T>{input_dims, input_strides}.generate(gen_value);

        output = tensor<T>{output_dims, output_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{output_dims, output_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_logsumexp_forward(input, ref_output, dims_vector);
        miopenStatus_t status;

        status = miopen::LogSumExpForward(handle,
                                          input.desc,
                                          input_dev.get(),
                                          output.desc,
                                          output_dev.get(),
                                          dims_vector.data(),
                                          dims_vector.size());

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;
        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold)
            << "Error output beyond tolerance Error: " << error << ",   Threshold " << threshold;
    }

    LogSumExpTestCase logsumexp_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;

    std::vector<int> dims_vector;
};

template <typename T = float>
struct LogSumExpBackwardTest : public ::testing::TestWithParam<LogSumExpTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle    = get_handle();
        logsumexp_config = GetParam();
        auto gen_value1   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2   = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 101); };

        dims_vector = logsumexp_config.reduce_dims;

        auto input_dims = logsumexp_config.input_size;
        auto input_grad_dims(input_dims);
        auto input_strides = ComputeStrides(input_dims, logsumexp_config.isContiguous);

        auto output_dims(input_dims);
        auto output_grad_dims(input_dims);
        for(const auto& dim : dims_vector)
        {
            output_dims[dim]      = 1;
            output_grad_dims[dim] = 1;
        }
        auto output_strides = ComputeStrides(output_dims, logsumexp_config.isContiguous);

        input       = tensor<T>{input_dims, input_strides}.generate(gen_value1);
        output      = tensor<T>{output_dims, output_strides};
        cpu_logsumexp_forward(input, output, dims_vector);

        output_grad = tensor<T>{output_grad_dims}.generate(gen_value2);

        input_grad = tensor<T>{input_grad_dims};
        std::fill(input_grad.begin(), input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        ref_input_grad = tensor<T>{input_grad_dims};
        std::fill(
            ref_input_grad.begin(), ref_input_grad.end(), std::numeric_limits<T>::quiet_NaN());

        input_dev       = handle.Write(input.data);
        input_grad_dev  = handle.Write(input_grad.data);
        output_dev      = handle.Write(output.data);
        output_grad_dev = handle.Write(output_grad.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_logsumexp_backward(
            input, ref_input_grad, output, output_grad, dims_vector.data(), dims_vector.size());
        miopenStatus_t status;

        status = miopen::LogSumExpBackward(handle,
                                           input.desc,
                                           input_dev.get(),
                                           output.desc,
                                           output_dev.get(),
                                           output_grad.desc,
                                           output_grad_dev.get(),
                                           input_grad.desc,
                                           input_grad_dev.get(),
                                           dims_vector.data(),
                                           dims_vector.size());

        EXPECT_EQ(status, miopenStatusSuccess);

        input_grad.data = handle.Read<T>(input_grad_dev, input_grad.data.size());
    }

    void Verify()
    {
        auto threshold = std::is_same<T, float>::value ? 1.5e-5 : 8.2e-2;

        if(std::is_same<T, bfloat16>::value)
            threshold *= 8.0;
        auto error = miopen::rms_range(ref_input_grad, input_grad);

        EXPECT_TRUE(miopen::range_distance(ref_input_grad) == miopen::range_distance(input_grad));
        EXPECT_TRUE(error < threshold) << "Error input_grad beyond tolerance Error: " << error
                                       << ",   Threshold " << threshold;
    }

    LogSumExpTestCase logsumexp_config;

    tensor<T> input;
    tensor<T> input_grad;
    tensor<T> output;
    tensor<T> output_grad;

    tensor<T> ref_input_grad;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr input_grad_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr output_grad_dev;

    std::vector<int> dims_vector;
};
