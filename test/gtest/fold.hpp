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
 * LIABILITY, WHETHER IN AN ACTN OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTN WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "cpu_unfold.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/fold.hpp>

struct FoldTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    std::vector<int32_t> outputSize;
    std::vector<int32_t> kernelSize;
    std::vector<int32_t> stride;
    std::vector<int32_t> padding;
    std::vector<int32_t> dilation;
    bool isContiguous = true;
    friend std::ostream& operator<<(std::ostream& os, const FoldTestCase& tc)
    {
        os << "N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H << " W:" << tc.W;
        os << " output_size:";
        for(const auto& outs : tc.outputSize)
            os << outs << " ";
        os << " kernel_size:";
        for(const auto& ks : tc.kernelSize)
            os << ks << " ";
        os << "stride:";
        for(const auto& s : tc.stride)
            os << s << " ";
        os << "padding:";
        for(const auto& p : tc.padding)
            os << p << " ";
        os << "dilation:";
        for(const auto& d : tc.dilation)
            os << d << " ";
        os << "isContiguous:" << std::boolalpha << tc.isContiguous;
        return os;
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
        else if((N != 0))
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }

    std::vector<size_t> ComputeStrides(std::vector<size_t> inputDim) const
    {
        if(!isContiguous)
            std::swap(inputDim.front(), inputDim.back());
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!isContiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

std::vector<FoldTestCase> FoldTestConfigs()
{
    // clang-format: off
    return {
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {4, 5}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, true},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {6, 11}, {2, 2}, {2, 3}, {0, 0}, {1, 1}, true},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {7, 12}, {2, 2}, {2, 3}, {0, 0}, {1, 1}, true},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {7, 13}, {2, 2}, {2, 3}, {0, 0}, {1, 1}, true},
        {3, 3 * 3 * 4, 0, 0, 3 * 4, {5, 7}, {3, 4}, {1, 1}, {0, 0}, {1, 1}, true},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {2, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, true},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {5, 7}, {2, 2}, {1, 1}, {0, 0}, {2, 3}, true},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {4, 5}, {2, 2}, {1, 1}, {0, 0}, {1, 1}, false},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {6, 11}, {2, 2}, {2, 3}, {0, 0}, {1, 1}, false},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {7, 12}, {2, 2}, {2, 3}, {0, 0}, {1, 1}, false},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {7, 13}, {2, 2}, {2, 3}, {0, 0}, {1, 1}, false},
        {3, 3 * 3 * 4, 0, 0, 3 * 4, {5, 7}, {3, 4}, {1, 1}, {0, 0}, {1, 1}, false},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {2, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1}, false},
        {3, 3 * 2 * 2, 0, 0, 3 * 4, {5, 7}, {2, 2}, {1, 1}, {0, 0}, {2, 3}, false},
    };
    // clang-format: on
}

template <typename T>
struct FoldFwdTest : public ::testing::TestWithParam<FoldTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        std::vector<size_t> in_dims    = config.GetInput();
        std::vector<size_t> in_strides = config.ComputeStrides(in_dims);

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_zero                 = [&](auto...) { return 0; };
        input                         = tensor<T>{in_dims, in_strides}.generate(gen_value);
        const int32_t N               = static_cast<int32_t>(in_dims[0]);
        int32_t C                     = static_cast<int32_t>(in_dims[1]);
        for(int32_t i : config.kernelSize)
        {
            C = C / i;
        }

        std::vector<size_t> out_dims{static_cast<size_t>(N),
                                     static_cast<size_t>(C),
                                     static_cast<size_t>(config.outputSize[0]),
                                     static_cast<size_t>(config.outputSize[1])};

        output     = tensor<T>{out_dims}.generate(gen_zero);
        outputHost = tensor<T>{out_dims}.generate(gen_zero);

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::FoldForward(handle,
                                     input.desc,
                                     input_dev.get(),
                                     output.desc,
                                     output_dev.get(),
                                     config.kernelSize.data(),
                                     static_cast<int>(config.kernelSize.size()),
                                     config.stride.data(),
                                     static_cast<int>(config.stride.size()),
                                     config.padding.data(),
                                     static_cast<int>(config.padding.size()),
                                     config.dilation.data(),
                                     static_cast<int>(config.dilation.size()));

        cpu_unfold_bwd_4d<T>(
            outputHost, input, config.kernelSize, config.stride, config.padding, config.dilation);

        EXPECT_EQ(status, miopenStatusSuccess);
        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        auto error_output = miopen::rms_range(outputHost, output);
        ASSERT_EQ(miopen::range_distance(outputHost), miopen::range_distance(output));

        EXPECT_LT(error_output, tolerance) << "Error forward output beyond tolerance Error: {"
                                           << error_output << "},  Tolerance: " << tolerance;
    }
    FoldTestCase config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> outputHost;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T>
struct FoldBwdTest : public ::testing::TestWithParam<FoldTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        std::vector<size_t> in_dims    = config.GetInput();
        std::vector<size_t> in_strides = config.ComputeStrides(in_dims);

        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_zero                 = [&](auto...) { return 0; };
        dinput                        = tensor<T>{in_dims, in_strides}.generate(gen_zero);
        dinputHost                    = tensor<T>{in_dims, in_strides}.generate(gen_zero);

        const int32_t N = static_cast<int32_t>(in_dims[0]);
        int32_t C       = static_cast<int32_t>(in_dims[1]);
        for(int32_t i : config.kernelSize)
        {
            C = C / i;
        }

        std::vector<size_t> out_dims{static_cast<size_t>(N),
                                     static_cast<size_t>(C),
                                     static_cast<size_t>(config.outputSize[0]),
                                     static_cast<size_t>(config.outputSize[1])};

        doutput = tensor<T>{out_dims}.generate(gen_value);

        dinput_dev  = handle.Write(dinput.data);
        doutput_dev = handle.Write(doutput.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::FoldBackward(handle,
                                      dinput.desc,
                                      dinput_dev.get(),
                                      doutput.desc,
                                      doutput_dev.get(),
                                      config.kernelSize.data(),
                                      static_cast<int>(config.kernelSize.size()),
                                      config.stride.data(),
                                      static_cast<int>(config.stride.size()),
                                      config.padding.data(),
                                      static_cast<int>(config.padding.size()),
                                      config.dilation.data(),
                                      static_cast<int>(config.dilation.size()));

        cpu_unfold_fwd_4d<T>(
            doutput, dinputHost, config.kernelSize, config.stride, config.padding, config.dilation);

        EXPECT_EQ(status, miopenStatusSuccess);
        dinput.data = handle.Read<T>(dinput_dev, dinput.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;
        auto error_dinput = miopen::rms_range(dinputHost, dinput);
        ASSERT_EQ(miopen::range_distance(dinputHost), miopen::range_distance(dinput));

        EXPECT_LT(error_dinput, tolerance) << "Error backward input_grad beyond tolerance Error: {"
                                           << error_dinput << "},  Tolerance: " << tolerance;
    }

    FoldTestCase config;

    tensor<T> dinput;
    tensor<T> doutput;

    tensor<T> dinputHost;

    miopen::Allocator::ManageDataPtr dinput_dev;
    miopen::Allocator::ManageDataPtr doutput_dev;
};
