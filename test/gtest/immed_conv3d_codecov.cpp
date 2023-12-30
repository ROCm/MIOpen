/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <gtest/gtest.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "test_env.hpp"

#include "conv3d.hpp"

MIOPEN_DECLARE_ENV_VAR_BOOL(CODECOV_TEST)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLAGS_ARGS)

namespace immed_conv3d_codecov {

template <class T>
struct conv3d_driver : conv_driver<T, ConvApi::Immediate>
{
    conv3d_driver() : conv_driver<T, ConvApi::Immediate>()
    {
        this->add(this->input_dims, "input");
        this->add(this->weight_tensor_dims, "weights");
        this->add(this->batch_size,
                  "batch_size",
                  this->generate_data_limited(this->get_batch_sizes(), 1, {8}));
        this->add(this->input_channels,
                  "input_channels",
                  this->generate_data_limited(this->get_input_channels(), 1, {2}));
        this->add(this->output_channels,
                  "output_channels",
                  this->generate_data_limited(this->get_output_channels(), 1, {16}));
        this->add(this->spatial_dim_elements,
                  "spatial_dim_elements",
                  this->generate_data_limited(this->get_3d_spatial_dims(), 1, {16, 16, 16}));
        this->add(this->filter_dims,
                  "filter_dims",
                  this->generate_data_limited(this->get_3d_filter_dims(), 2, {5, 5, 5}));
        this->add(this->pads_strides_dilations,
                  "pads_strides_dilations",
                  this->generate_data_limited(this->get_3d_pads_strides_dilations(), 2));
        this->add(this->trans_output_pads,
                  "trans_output_pads",
                  this->generate_data_limited(this->get_3d_trans_output_pads(), 1));
        this->add(this->in_layout, "in_layout", this->generate_data({"NCDHW"}));
        this->add(this->fil_layout, "fil_layout", this->generate_data({"NCDHW"}));
        this->add(this->out_layout, "out_layout", this->generate_data({"NCDHW"}));
    }
};

static bool SkipTest(void) { return !miopen::IsEnabled(ENV(CODECOV_TEST)); }

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

void Run3dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenHalf: params = Conv3dHalf::GetParam(); break;
    case miopenBFloat16: params = Conv3dBFloat16::GetParam(); break;
    case miopenFloat: params = Conv3dFloat::GetParam(); break;
    case miopenInt8: params = Conv3dInt8::GetParam(); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt32:
    case miopenDouble:
        FAIL() << "miopenInt32, miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data type not supported by "
                  "immed_conv3d_codecov test";

    default: params = Conv3dFloat::GetParam();
    }

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<conv3d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

bool IsTestSupportedForDevice(const miopen::Handle& handle) { return true; }

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const auto& flag_arg = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLAGS_ARGS));

    const std::vector<std::string> test_cases = {
        // clang-format off
    {"test_immed_conv3d " + precision + " --input 1 4 4 4 4 --weights 2 4 3 3 3 --pads_strides_dilations 0 0 0 1 1 1 1 1 1 "+flag_arg}
        // clang-format on
    };

    return test_cases;
}

} // namespace immed_conv3d_codecov
using namespace immed_conv3d_codecov;

TEST_P(Conv3dFloat, FloatTest_immed_conv3d_codecov)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--float"))
    {
        Run3dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv3dHalf, HalfTest_immed_conv3d_codecov)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--half"))
    {
        Run3dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv3dBFloat16, BFloat16Test_immed_conv3d_codecov)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--bfloat16"))
    {
        Run3dDriver(miopenBFloat16);
    }
    else
    {
        GTEST_SKIP();
    }
};

TEST_P(Conv3dInt8, Int8Test_immed_conv3d_codecov)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && !SkipTest() && IsTestRunWith("--int8"))
    {
        Run3dDriver(miopenInt8);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(ImmedConv3D, Conv3dFloat, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(ImmedConv3D, Conv3dHalf, testing::Values(GetTestCases("--half")));

INSTANTIATE_TEST_SUITE_P(ImmedConv3D, Conv3dBFloat16, testing::Values(GetTestCases("--bfloat16")));

INSTANTIATE_TEST_SUITE_P(ImmedConv3D, Conv3dInt8, testing::Values(GetTestCases("--int8")));
