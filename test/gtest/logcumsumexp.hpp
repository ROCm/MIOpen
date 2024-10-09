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

#include "cpu_logcumsumexp.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/logcumsumexp.hpp>
#include <miopen/logcumsumexp/solvers.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

struct LogCumSumExpTestCase
{
    std::vector<size_t> lengths;
    int dim;
    bool exclusive;
    bool reverse;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os, const LogCumSumExpTestCase& tc)
    {
        return os << " Lengths:" << tc.lengths << " Dim:" << tc.dim
                  << " Exclusive:" << (tc.exclusive ? "True" : "False")
                  << " Reverse:" << (tc.reverse ? "True" : "False")
                  << " Contiguous:" << (tc.contiguous ? "True" : "False");
    }
};

inline std::vector<size_t> GetStrides(std::vector<size_t> lengths, bool contiguous)
{
    if(!contiguous)
        std::swap(lengths.front(), lengths.back());
    std::vector<size_t> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

inline std::vector<LogCumSumExpTestCase>
LogCumSumExpTestConfigs(const std::vector<std::vector<size_t>>& SizeList)
{
    std::vector<LogCumSumExpTestCase> tcs;

    std::vector<size_t> dims      = {-1, 0, 1};
    std::vector<bool> exclusives  = {false, true};
    std::vector<bool> reverses    = {false, true};
    std::vector<bool> contiguouss = {true, false};

    auto&& handle = get_handle();
    for(const auto& lengths : SizeList)
    {
        auto out_strides = GetStrides(lengths, true);
        for(auto contiguous : contiguouss)
        {
            auto input_strides = GetStrides(lengths, contiguous);
            for(auto dim : dims)
            {
                for(auto exclusive : exclusives)
                {
                    for(auto reverse : reverses)
                    {
                        if(miopen::solver::logcumsumexp::ForwardContiguousSmallCumDimStride1()
                               .IsApplicable(miopen::ExecutionContext(&handle),
                                             miopen::logcumsumexp::ForwardProblemDescription(
                                                 miopen::TensorDescriptor(
                                                     miopen_type<float>{}, lengths, input_strides),
                                                 miopen::TensorDescriptor(
                                                     miopen_type<float>{}, lengths, out_strides),
                                                 dim)) ||
                           miopen::solver::logcumsumexp::ForwardSmallCumDim().IsApplicable(
                               miopen::ExecutionContext(&handle),
                               miopen::logcumsumexp::ForwardProblemDescription(
                                   miopen::TensorDescriptor(
                                       miopen_type<float>{}, lengths, input_strides),
                                   miopen::TensorDescriptor(
                                       miopen_type<float>{}, lengths, out_strides),
                                   dim)))
                            tcs.push_back({lengths, dim, exclusive, reverse, contiguous});
                    }
                }
            }
        }
    }

    return tcs;
}

inline std::vector<std::vector<size_t>> GetSmokeTestSize()
{
    return {
        {1, 10},
        {65, 100},
        {1, 65},
        {70, 10},
    };
}

inline std::vector<std::vector<size_t>> GetSmokePerfSize()
{
    return {
        {512, 64, 112},
        {1024, 7, 7, 1024},
    };
}

inline std::vector<LogCumSumExpTestCase> LogCumSumExpSmokeTestConfigs()
{
    return LogCumSumExpTestConfigs(GetSmokeTestSize());
}

inline std::vector<LogCumSumExpTestCase> LogCumSumExpPerfTestConfigs()
{
    return LogCumSumExpTestConfigs(GetSmokePerfSize());
}

inline std::vector<LogCumSumExpTestCase> LogCumSumExpFullTestConfigs()
{
    std::vector<LogCumSumExpTestCase> tcs;

    auto smoke_test = LogCumSumExpSmokeTestConfigs();
    auto perf_test  = LogCumSumExpPerfTestConfigs();

    tcs.reserve(smoke_test.size() + perf_test.size());
    for(const auto& test : smoke_test)
        tcs.push_back(test);
    for(const auto& test : perf_test)
        tcs.push_back(test);

    return tcs;
}

template <typename T>
struct LogCumSumExpTestFwd : public ::testing::TestWithParam<LogCumSumExpTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle       = get_handle();
        logcumsumexp_config = GetParam();
        auto gen_value      = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        auto lengths = logcumsumexp_config.lengths;

        auto input_strides = GetStrides(lengths, logcumsumexp_config.contiguous);
        input              = tensor<T>{lengths, input_strides}.generate(gen_value);

        output     = tensor<T>{lengths};
        ref_output = tensor<T>{lengths};

        input_dev  = handle.Write(input.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        cpu_logcumsumexp_forward<T>(input,
                                    ref_output,
                                    logcumsumexp_config.dim,
                                    logcumsumexp_config.exclusive,
                                    logcumsumexp_config.reverse);

        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::logcumsumexp::LogCumSumExpForward(handle,
                                                           input.desc,
                                                           input_dev.get(),
                                                           output.desc,
                                                           output_dev.get(),
                                                           logcumsumexp_config.dim,
                                                           logcumsumexp_config.exclusive,
                                                           logcumsumexp_config.reverse);
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

        auto error_output = miopen::rms_range(ref_output, output);
        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error_output, tolerance)
            << "Error forward Output beyond tolerance Error: " << error_output
            << " Tolerance: " << tolerance;
    }

    LogCumSumExpTestCase logcumsumexp_config;

    tensor<T> input;
    tensor<T> output;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T>
struct LogCumSumExpTestBwd : public ::testing::TestWithParam<LogCumSumExpTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        logcumsumexp_config  = GetParam();
        auto gen_value_input = [](auto...) {
            return prng::gen_descreet_uniform_sign<T>(1e-2, 100);
        };
        auto gen_value_doutput = [](auto...) {
            return prng::gen_descreet_uniform_sign<T>(1e-2, 100);
        };

        auto lengths = logcumsumexp_config.lengths;

        auto input_strides = GetStrides(lengths, logcumsumexp_config.contiguous);
        input              = tensor<T>{lengths, input_strides}.generate(gen_value_input);
        dinput             = tensor<T>{lengths, input_strides};

        output  = tensor<T>{lengths};
        doutput = tensor<T>{lengths}.generate(gen_value_doutput);

        // Calculate output tensor value by forwarding input tensor
        cpu_logcumsumexp_forward(input,
                                 output,
                                 logcumsumexp_config.dim,
                                 logcumsumexp_config.exclusive,
                                 logcumsumexp_config.reverse);

        ref_dinput = tensor<T>{lengths, input_strides};

        input_dev   = handle.Write(input.data);
        output_dev  = handle.Write(output.data);
        doutput_dev = handle.Write(doutput.data);
        dinput_dev  = handle.Write(dinput.data);
    }

    void RunTest()
    {
        cpu_logcumsumexp_backward<T>(input,
                                     output,
                                     doutput,
                                     ref_dinput,
                                     logcumsumexp_config.dim,
                                     logcumsumexp_config.exclusive,
                                     logcumsumexp_config.reverse);

        auto&& handle = get_handle();
        auto status   = miopen::logcumsumexp::LogCumSumExpBackward(handle,
                                                                 input.desc,
                                                                 input_dev.get(),
                                                                 output.desc,
                                                                 output_dev.get(),
                                                                 doutput.desc,
                                                                 doutput_dev.get(),
                                                                 dinput.desc,
                                                                 dinput_dev.get(),
                                                                 logcumsumexp_config.dim,
                                                                 logcumsumexp_config.exclusive,
                                                                 logcumsumexp_config.reverse);
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

        auto error_dinput = miopen::rms_range(ref_dinput, dinput);
        ASSERT_EQ(miopen::range_distance(ref_dinput), miopen::range_distance(dinput));
        EXPECT_LT(error_dinput, tolerance)
            << "Error backward Input Gradient beyond tolerance Error: " << error_dinput
            << " Tolerance: " << tolerance;
    }

    LogCumSumExpTestCase logcumsumexp_config;

    tensor<T> input;
    tensor<T> output;
    tensor<T> doutput;
    tensor<T> dinput;

    tensor<T> ref_dinput;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr doutput_dev;
    miopen::Allocator::ManageDataPtr dinput_dev;
};
