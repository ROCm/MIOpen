#include <tuple>

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/env.hpp>
#include "conv_common.hpp"
#include "get_handle.hpp"

std::string GetFloatArg()
{
    static const auto tmp = miopen::GetEnv("MIOPEN_TEST_FLOAT_ARG");
    if(tmp.empty())
    {
        return "";
    }
    return tmp.front();
};

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class ConfigWithHalf : public testing::TestWithParam<std::vector<std::string>>
{
};
class ConfigWithInt8 : public testing::TestWithParam<std::vector<std::string>>
{
};
class ConfigWithBFloat16 : public testing::TestWithParam<std::vector<std::string>>
{
};
class ConfigWithFloat : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenFloat: params = ConfigWithFloat::GetParam(); break;
    case miopenHalf: params = ConfigWithHalf::GetParam(); break;
    case miopenInt8: params = ConfigWithInt8::GetParam(); break;
    case miopenBFloat16: params = ConfigWithBFloat16::GetParam(); break;
    case miopenInt8x4:
    case miopenInt32:
    case miopenDouble:
        MIOPEN_THROW(miopenStatusBadParm,
                     "miopenInt8x4, miopenInt32, miopenDouble data type not supported by "
                     "conv_embed_db test");

    default: params = ConfigWithFloat::GetParam();
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
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        EXPECT_FALSE(capture.find("Perf Db: record not found") != std::string::npos);
    }
};

TEST_P(ConfigWithFloat, FloatTest)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(miopen::StartsWith(handle.GetDeviceName(), "gfx908") || // Explicitly disabled
       miopen::StartsWith(handle.GetDeviceName(), "gfx90a") || // Explicitly disabled
       miopen::StartsWith(handle.GetDeviceName(), "gfx94") ||  // Implicitly disabled by default
       miopen::StartsWith(handle.GetDeviceName(), "gfx103") || // Implicitly disabled by default
       miopen::StartsWith(handle.GetDeviceName(), "gfx110") || // Implicitly disabled by default
       GetFloatArg() != "--float")
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenFloat);
    }

#else
    GTEST_SKIP();
#endif
};

TEST_P(ConfigWithHalf, HalfTest)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(miopen::StartsWith(handle.GetDeviceName(), "gfx908") || // Explicitly disabled
       miopen::StartsWith(handle.GetDeviceName(), "gfx90a") || // Explicitly disabled
       miopen::StartsWith(handle.GetDeviceName(), "gfx94") ||  // Implicitly disabled by default
       miopen::StartsWith(handle.GetDeviceName(), "gfx103") || // Implicitly disabled by default
       miopen::StartsWith(handle.GetDeviceName(), "gfx110") || // Implicitly disabled by default
       GetFloatArg() != "--half")
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenHalf);
    }

#else
    GTEST_SKIP();
#endif
};

TEST_P(ConfigWithInt8, Int8Test)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(miopen::StartsWith(handle.GetDeviceName(), "gfx908") || // Explicitly disabled
       miopen::StartsWith(handle.GetDeviceName(), "gfx90a") || // Explicitly disabled
       miopen::StartsWith(handle.GetDeviceName(), "gfx94") ||  // Implicitly disabled by default
       miopen::StartsWith(handle.GetDeviceName(), "gfx103") || // Implicitly disabled by default
       miopen::StartsWith(handle.GetDeviceName(), "gfx110") || // Implicitly disabled by default
       GetFloatArg() != "--int8")
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenInt8);
    }

#else
    GTEST_SKIP();
#endif
};

TEST_P(ConfigWithBFloat16, BFloat16Test)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(miopen::StartsWith(handle.GetDeviceName(), "gfx908") || // Explicitly disabled
       miopen::StartsWith(handle.GetDeviceName(), "gfx90a") || // Explicitly disabled
       miopen::StartsWith(handle.GetDeviceName(), "gfx94") ||  // Implicitly disabled by default
       miopen::StartsWith(handle.GetDeviceName(), "gfx103") || // Implicitly disabled by default
       miopen::StartsWith(handle.GetDeviceName(), "gfx110") || // Implicitly disabled by default
       GetFloatArg() != "--bfloat16")
    {
        GTEST_SKIP();
    }
    else
    {
        Run2dDriver(miopenBFloat16);
    }

#else
    GTEST_SKIP();
#endif
};

std::vector<std::string> GetTestCases(const std::string& precision)
{
    std::string flags = " --disable-validation --verbose ";

    const std::vector<std::string> test_cases = {
        // clang-format off
    {precision + flags + "--input 128 128 28 28 --weights 128 128 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    {precision + flags + "--input 128 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {precision + flags + "--input 128 3 230 230   --weights 64 3 7 7 --pads_strides_dilations 0 0 2 2 1 1"},
    {precision + flags + "--input 128 64 56 56 --weights 64 64 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    {precision + flags + "--input 128 256 14 14 --weights 256 256 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    {precision + flags + "--input 128 512 7 7   --weights 512 512 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    {precision + flags + "--input 128 1024 14 14 --weights 512 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {precision + flags + "--input 128 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {precision + flags + "--input 128 256 14 14 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {precision + flags + "--input 128 512 28 28 --weights 256 512 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {precision + flags + "--input 128 1024 14 14 --weights 256 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {precision + flags + "--input 128 64 56 56 --weights 256 64 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {precision + flags + "--input 128 64 56 56 --weights 64 64 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {precision + flags + "--input 128 128 28 28 --weights 512 128 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {precision + flags + "--input 128 256 56 56 --weights 128 256 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {precision + flags + "--input 128 256 56 56 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {precision + flags + "--input 128 512 28 28 --weights 1024 512 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    {precision + flags + "--input 128 512 28 28 --weights 128 512 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {precision + flags + "--input 128 512 7 7   --weights 2048 512 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    {precision + flags + "--input 128 2048 7 7 --weights 512 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1"}
        // clang-format on
    };

    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(ConvEmbedDB, ConfigWithFloat, testing::Values(GetTestCases("--float")));
INSTANTIATE_TEST_SUITE_P(ConvEmbedDB, ConfigWithHalf, testing::Values(GetTestCases("--half")));
INSTANTIATE_TEST_SUITE_P(ConvEmbedDB, ConfigWithInt8, testing::Values(GetTestCases("--int8")));
INSTANTIATE_TEST_SUITE_P(ConvEmbedDB,
                         ConfigWithBFloat16,
                         testing::Values(GetTestCases("--bfloat16")));
