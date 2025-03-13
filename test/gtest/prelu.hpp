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

#include "cpu_prelu.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"

#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/prelu.hpp>

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

struct PReLUTestCase
{
    std::vector<size_t> lengths;
    bool full_params;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os, const PReLUTestCase& tc)
    {
        return os << " Lengths:" << tc.lengths
                  << " Full_num_params:" << (tc.full_params ? "True" : "False")
                  << " Contiguous:" << (tc.contiguous ? "True" : "False");
    }
};

inline std::vector<PReLUTestCase> PReLUSmokeTestConfigs()
{
    std::vector<PReLUTestCase> tcs;

    tcs.push_back({{64, 4}, true, true});
    tcs.push_back({{64, 4}, true, false});
    tcs.push_back({{64, 4}, false, true});
    tcs.push_back({{64, 4}, false, false});

    tcs.push_back({{64, 112}, true, true});
    tcs.push_back({{64, 112}, true, false});
    tcs.push_back({{64, 112}, false, true});
    tcs.push_back({{64, 112}, false, false});

    return tcs;
}

inline std::vector<PReLUTestCase> PReLUPerfTestConfigs()
{
    std::vector<PReLUTestCase> tcs;

    tcs.push_back({{64, 112, 50}, true, true});
    tcs.push_back({{64, 112, 50}, true, false});
    tcs.push_back({{64, 112, 50}, false, true});
    tcs.push_back({{64, 112, 50}, false, false});

    return tcs;
}

inline std::vector<PReLUTestCase> PReLUFullTestConfigs()
{
    std::vector<PReLUTestCase> tcs;

    auto smoke_test = PReLUSmokeTestConfigs();
    auto perf_test  = PReLUPerfTestConfigs();

    tcs.reserve(smoke_test.size() + perf_test.size());
    for(const auto& test : smoke_test)
        tcs.push_back(test);
    for(const auto& test : perf_test)
        tcs.push_back(test);

    return tcs;
}

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

template <typename T = float>
struct PReLUTest : public ::testing::TestWithParam<PReLUTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        prelu_config    = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-6, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-6, 99); };

        auto lengths = prelu_config.lengths;

        auto input_strides = GetStrides(lengths, prelu_config.contiguous);
        input              = tensor<T>{lengths, input_strides}.generate(gen_value1);

        std::vector<size_t> weight_length = {1};
        if(prelu_config.full_params)
            weight_length = {lengths[1]};
        std::vector<size_t> weight_strides = {prelu_config.contiguous ? 1 : 2};
        weight = tensor<T>{weight_length, weight_strides}.generate(gen_value2);

        ws_sizeInBytes = miopen::GetPReLUBackwardWorkspaceSize(handle, input.desc, weight.desc);
        if(ws_sizeInBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(ws_sizeInBytes != 0)
        {
            std::vector<size_t> workspace_dims;
            workspace_dims.push_back(ws_sizeInBytes / sizeof(float));

            workspace = tensor<float>{workspace_dims};
            std::fill(workspace.begin(), workspace.end(), 0.0f);

            workspace_dev = handle.Write(workspace.data);
        }

        dinput     = tensor<T>{lengths, input_strides};
        ref_dinput = tensor<T>{lengths, input_strides};
        std::fill(dinput.begin(), dinput.end(), static_cast<T>(0.0f));
        std::fill(ref_dinput.begin(), ref_dinput.end(), static_cast<T>(0.0f));

        dweight     = tensor<T>{weight_length, weight_strides};
        ref_dweight = tensor<T>{weight_length, weight_strides};
        std::fill(dweight.begin(), dweight.end(), static_cast<T>(0.0f));
        std::fill(ref_dweight.begin(), ref_dweight.end(), static_cast<T>(0.0f));

        auto out_strides = GetStrides(lengths, true);
        doutput          = tensor<T>{lengths, out_strides};
        std::fill(doutput.begin(), doutput.end(), static_cast<T>(1.0f));

        input_dev   = handle.Write(input.data);
        weight_dev  = handle.Write(weight.data);
        doutput_dev = handle.Write(doutput.data);
        dinput_dev  = handle.Write(dinput.data);
        dweight_dev = handle.Write(dweight.data);
    }

    void RunTest()
    {
        cpu_prelu_backward<T>(input, weight, doutput, ref_dinput, ref_dweight);

        auto&& handle = get_handle();
        miopenStatus_t status;

        status = miopen::PReLUBackward(handle,
                                       workspace_dev.get(),
                                       ws_sizeInBytes,
                                       input.desc,
                                       input_dev.get(),
                                       weight.desc,
                                       weight_dev.get(),
                                       doutput.desc,
                                       doutput_dev.get(),
                                       dinput.desc,
                                       dinput_dev.get(),
                                       dweight.desc,
                                       dweight_dev.get());
        EXPECT_EQ(status, miopenStatusSuccess);
        dinput.data  = handle.Read<T>(dinput_dev, dinput.data.size());
        dweight.data = handle.Read<T>(dweight_dev, dweight.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;

        auto error_dinput  = miopen::rms_range(ref_dinput, dinput);
        auto error_dweight = miopen::rms_range(ref_dweight, dweight);
        ASSERT_EQ(miopen::range_distance(ref_dinput), miopen::range_distance(dinput));
        ASSERT_EQ(miopen::range_distance(ref_dweight), miopen::range_distance(dweight));
        EXPECT_LT(error_dinput, tolerance)
            << "Error backward Input Gradient beyond tolerance Error: " << error_dinput
            << ",  Tolerance: " << tolerance;
        EXPECT_LT(error_dweight, tolerance)
            << "Error backward Weight Gradient beyond tolerance Error: " << error_dweight
            << ",  Tolerance: " << tolerance;
    }

    PReLUTestCase prelu_config;

    tensor<T> input;
    tensor<T> weight;
    tensor<T> doutput;
    tensor<T> dinput;
    tensor<T> dweight;
    tensor<float> workspace;

    tensor<T> ref_dinput;
    tensor<T> ref_dweight;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr weight_dev;
    miopen::Allocator::ManageDataPtr doutput_dev;
    miopen::Allocator::ManageDataPtr dinput_dev;
    miopen::Allocator::ManageDataPtr dweight_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;
};
