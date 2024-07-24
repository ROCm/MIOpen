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
#include <tuple>

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/env.hpp>
#include "get_handle.hpp"
#include "test_env.hpp"

#include "../conv2d.hpp"

namespace conv_embed_db {

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class ConvEmbedConfigHalf : public testing::TestWithParam<std::vector<std::string>>
{
};
class ConvEmbedConfigInt8 : public testing::TestWithParam<std::vector<std::string>>
{
};
class ConvEmbedConfigBFloat16 : public testing::TestWithParam<std::vector<std::string>>
{
};
class ConvEmbedConfigFloat : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = ConvEmbedConfigFloat::GetParam(); break;
    case miopenHalf: params = ConvEmbedConfigHalf::GetParam(); break;
    case miopenInt8: params = ConvEmbedConfigInt8::GetParam(); break;
    case miopenBFloat16: params = ConvEmbedConfigBFloat16::GetParam(); break;
    case miopenInt64:
    case miopenInt32:
    case miopenFloat8:
    case miopenBFloat8:
    case miopenDouble:
        FAIL() << "miopenInt32, miopenFloat8, miopenBFloat8, miopenDouble data type "
                  "not supported by conv_embed_db test";

    default: params = ConvEmbedConfigFloat::GetParam();
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
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data(), "test_conv_embed_db");
        auto capture = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(capture.find("Perf Db: record not found") != std::string::npos);
    }
};

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx900" || devName == "gfx906")
        return true;
    else
        return false;
}

std::vector<std::string> GetTestCases(const std::string& precision)
{
    std::string flags = precision + " --disable-validation --verbose ";

    const std::vector<std::string> test_cases = {
        // clang-format off
    {flags + "--input 128 128 28 28 --weights 128 128 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    {flags + "--input 128 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {flags + "--input 128 3 230 230   --weights 64 3 7 7 --pads_strides_dilations 0 0 2 2 1 1"},
    {flags + "--input 128 64 56 56 --weights 64 64 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    {flags + "--input 128 256 14 14 --weights 256 256 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    {flags + "--input 128 512 7 7   --weights 512 512 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    {flags + "--input 128 1024 14 14 --weights 512 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {flags + "--input 128 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {flags + "--input 128 256 14 14 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {flags + "--input 128 512 28 28 --weights 256 512 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {flags + "--input 128 1024 14 14 --weights 256 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {flags + "--input 128 64 56 56 --weights 256 64 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {flags + "--input 128 64 56 56 --weights 64 64 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {flags + "--input 128 128 28 28 --weights 512 128 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {flags + "--input 128 256 56 56 --weights 128 256 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {flags + "--input 128 256 56 56 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {flags + "--input 128 512 28 28 --weights 1024 512 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {flags + "--input 128 512 28 28 --weights 128 512 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {flags + "--input 128 512 7 7   --weights 2048 512 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {flags + "--input 128 2048 7 7 --weights 512 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1"}
        // clang-format on
    };

    return test_cases;
}

} // namespace conv_embed_db
using namespace conv_embed_db;

TEST_P(ConvEmbedConfigFloat, FloatTest_conv_embed_db)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && IsTestRunWith("--float"))
    {
        Run2dDriver(miopenFloat);
    }
    else
    {
        GTEST_SKIP();
    }

#else
    GTEST_SKIP();
#endif
};

TEST_P(ConvEmbedConfigHalf, HalfTest_conv_embed_db)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && IsTestRunWith("--half"))
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }

#else
    GTEST_SKIP();
#endif
};

TEST_P(ConvEmbedConfigInt8, Int8Test_conv_embed_db)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && IsTestRunWith("--int8"))
    {
        Run2dDriver(miopenInt8);
    }
    else
    {
        GTEST_SKIP();
    }

#else
    GTEST_SKIP();
#endif
};

TEST_P(ConvEmbedConfigBFloat16, BFloat16Test_conv_embed_db)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle) && IsTestRunWith("--bfloat16"))
    {
        Run2dDriver(miopenBFloat16);
    }
    else
    {
        GTEST_SKIP();
    }

#else
    GTEST_SKIP();
#endif
};

INSTANTIATE_TEST_SUITE_P(ConvEmbedDB,
                         ConvEmbedConfigFloat,
                         testing::Values(GetTestCases("--float")));
INSTANTIATE_TEST_SUITE_P(ConvEmbedDB, ConvEmbedConfigHalf, testing::Values(GetTestCases("--half")));
INSTANTIATE_TEST_SUITE_P(ConvEmbedDB, ConvEmbedConfigInt8, testing::Values(GetTestCases("--int8")));
INSTANTIATE_TEST_SUITE_P(ConvEmbedDB,
                         ConvEmbedConfigBFloat16,
                         testing::Values(GetTestCases("--bfloat16")));
