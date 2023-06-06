#include <tuple>

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/env.hpp>
#include "conv_2d.hpp"
#include "get_handle.hpp"

using TestCase = std::tuple<std::vector<std::string>, std::string>;

std::string GetFloatArg()
{
    static const auto tmp = miopen::GetEnv("MIOPEN_TEST_FLOAT_ARG");
    if(tmp.empty())
    {
        return "";
    }
    return tmp.front();
};

std::vector<std::string> GetEnvVars(const std::vector<std::string>& check_vars)
{
    std::vector<std::string> vars = {};
    for(const auto& cvar : check_vars)
    {
        static const auto tmp = miopen::GetEnv(cvar.c_str());
        if(!tmp.empty())
        {
            vars.push_back(cvar + "=0");
        }
    }
    return vars;
};

void GetArgs(const TestCase& param, std::vector<std::string>& tokens)
{
    auto env_vars = std::get<0>(param);
    for(auto& elem : env_vars)
    {
        putenv(elem.data());
    }

    auto cmd = std::get<1>(param);

    std::stringstream ss(cmd);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class Conv2dHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};
class Conv2dInt8 : public testing::TestWithParam<std::vector<TestCase>>
{
};
class Conv2dBFloat16 : public testing::TestWithParam<std::vector<TestCase>>
{
};
class Conv2dFloat : public testing::TestWithParam<std::vector<TestCase>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<TestCase> params;
    switch(prec)
    {
    case miopenFloat: params = Conv2dFloat::GetParam(); break;
    case miopenHalf: params = Conv2dHalf::GetParam(); break;
    case miopenInt8: params = Conv2dInt8::GetParam(); break;
    case miopenBFloat16: params = Conv2dBFloat16::GetParam(); break;
    case miopenInt8x4:
    case miopenInt32:
    case miopenDouble:
        MIOPEN_THROW(miopenStatusBadParm,
                     "miopenInt8x4, miopenInt32, miopenDouble data type not supported by "
                     "conv_embed_db test");

    default: params = Conv2dFloat::GetParam();
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

TEST_P(Conv2dFloat, FloatTest)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(miopen::StartsWith(handle.GetDeviceName(), "gfx908") ||
       miopen::StartsWith(handle.GetDeviceName(), "gfx90a") || GetFloatArg() != "--float")
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

TEST_P(Conv2dHalf, HalfTest)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(miopen::StartsWith(handle.GetDeviceName(), "gfx908") ||
       miopen::StartsWith(handle.GetDeviceName(), "gfx90a") || GetFloatArg() != "--half")
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

TEST_P(Conv2dInt8, Int8Test)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(miopen::StartsWith(handle.GetDeviceName(), "gfx908") ||
       miopen::StartsWith(handle.GetDeviceName(), "gfx90a") || GetFloatArg() != "--int8")
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

TEST_P(Conv2dBFloat16, BFloat16Test)
{
#if MIOPEN_EMBED_DB

    const auto& handle = get_handle();
    if(miopen::StartsWith(handle.GetDeviceName(), "gfx908") ||
       miopen::StartsWith(handle.GetDeviceName(), "gfx90a") || GetFloatArg() != "--bfloat16")
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

std::vector<TestCase> GetTestCases(const std::string& precision)
{
    std::vector<std::string> winograd = {
        "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2"}; // WORKAROUND for issue #1008 when set to 0
    std::vector<std::string> igemm_wrw = {
        "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2",           // WORKAROUND for issue #1008 when set to 0
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_4R1"}; // WORKAROUND for issue #874 when set to 0
    std::vector<std::string> igemm_fwd = {
        "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2",           // WORKAROUND for issue #1008 when set to 0
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_4R1"}; // WORKAROUND for issue #874 when set to 0
    std::vector<std::string> igemm_fwd_wrw = {
        "MIOPEN_DEBUG_AMD_WINOGRAD_RXS_F3X2",           // WORKAROUND for issue #1008 when set to 0
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_4R1",  // WORKAROUND for issue #874 when set to 0
        "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_WRW_4R1"}; // WORKAROUND for issue #874 when set to 0

    std::string flags                      = " --disable-validation --verbose";
    const std::vector<TestCase> test_cases = {
        // clang-format off
    TestCase{GetEnvVars(winograd), precision + flags + " --input 128 128 28 28 --weights 128 128 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    TestCase{GetEnvVars(winograd), precision + flags + " --input 128 256 56 56 --weights 512 256 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{GetEnvVars(winograd), precision + flags + " --input 128 3 230 230   --weights 64 3 7 7 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{GetEnvVars(igemm_wrw), precision + flags + " --input 128 64 56 56 --weights 64 64 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    TestCase{GetEnvVars(igemm_wrw), precision + flags + " --input 128 256 14 14 --weights 256 256 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    TestCase{GetEnvVars(igemm_wrw), precision + flags + " --input 128 512 7 7   --weights 512 512 3 3 --pads_strides_dilations 1 1 1 1 1 1"},
    TestCase{GetEnvVars(igemm_wrw), precision + flags + " --input 128 1024 14 14 --weights 512 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{GetEnvVars(igemm_wrw), precision + flags + " --input 128 1024 14 14 --weights 2048 1024 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{GetEnvVars(igemm_fwd), precision + flags + " --input 128 256 14 14 --weights 1024 256 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{GetEnvVars(igemm_fwd), precision + flags + " --input 128 512 28 28 --weights 256 512 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{GetEnvVars(igemm_fwd), precision + flags + " --input 128 1024 14 14 --weights 256 1024 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{GetEnvVars(igemm_fwd_wrw), precision + flags + " --input 128 64 56 56 --weights 256 64 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{GetEnvVars(igemm_fwd_wrw), precision + flags + " --input 128 64 56 56 --weights 64 64 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{GetEnvVars(igemm_fwd_wrw), precision + flags + " --input 128 128 28 28 --weights 512 128 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{GetEnvVars(igemm_fwd_wrw), precision + flags + " --input 128 256 56 56 --weights 128 256 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{GetEnvVars(igemm_fwd_wrw), precision + flags + " --input 128 256 56 56 --weights 64 256 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{GetEnvVars(igemm_fwd_wrw), precision + flags + " --input 128 512 28 28 --weights 1024 512 1 1 --pads_strides_dilations 0 0 2 2 1 1"},
    TestCase{GetEnvVars(igemm_fwd_wrw), precision + flags + " --input 128 512 28 28 --weights 128 512 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{GetEnvVars(igemm_fwd_wrw), precision + flags + " --input 128 512 7 7   --weights 2048 512 1 1 --pads_strides_dilations 0 0 1 1 1 1"},
    TestCase{GetEnvVars(igemm_fwd_wrw), precision + flags + " --input 128 2048 7 7 --weights 512 2048 1 1 --pads_strides_dilations 0 0 1 1 1 1"}
        // clang-format on
    };

    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(Conv2dGroup, Conv2dFloat, testing::Values(GetTestCases("--float")));
INSTANTIATE_TEST_SUITE_P(Conv2dGroup, Conv2dHalf, testing::Values(GetTestCases("--half")));
INSTANTIATE_TEST_SUITE_P(Conv2dGroup, Conv2dInt8, testing::Values(GetTestCases("--int8")));
INSTANTIATE_TEST_SUITE_P(Conv2dGroup, Conv2dBFloat16, testing::Values(GetTestCases("--bfloat16")));
