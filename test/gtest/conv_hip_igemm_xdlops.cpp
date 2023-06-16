#include <tuple>

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/env.hpp>
#include "conv_2d.hpp"
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

class ConfigWithInt8 : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{
    std::vector<std::string> params;
    switch(prec)
    {
    case miopenInt8: params = ConfigWithInt8::GetParam(); break;
    case miopenHalf:
    case miopenBFloat16:
    case miopenFloat:
    case miopenInt8x4:
    case miopenInt32:
    case miopenDouble:
        MIOPEN_THROW(
            miopenStatusBadParm,
            "miopenHalf, miopenBFloat16, miopenFloat, miopenInt8x4, miopenInt32, miopenDouble data "
            "type not supported by "
            "conv_hip_igemm_mlir_xdlops test");

    default: params = ConfigWithInt8::GetParam();
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

TEST_P(ConfigWithInt8, Int8Test)
{
#if MIOPEN_BACKEND_OPENCL

    GTEST_SKIP() << "MIOPEN_BACKEND_HIP needed for this test";

#else // MIOPEN_BACKEND_HIP, OCL_DISABLED
    const auto& handle = get_handle();
    if((miopen::StartsWith(handle.GetDeviceName(), "gfx908") ||
        miopen::StartsWith(handle.GetDeviceName(), "gfx90a") ||
        miopen::StartsWith(handle.GetDeviceName(), "gfx94")) &&
       miopen::IsEnvvarValueEnabled("MIOPEN_TEST_COMPOSABLEKERNEL") &&
       miopen::IsEnvvarValueEnabled("MIOPEN_TEST_ALL") && GetFloatArg() == "--int8")
    {
        Run2dDriver(miopenInt8);
    }
    else
    {
        GTEST_SKIP();
    }
#endif
};

std::vector<std::string> GetTestCases(const std::string& precision)
{
    std::string fwd         = " --disable-backward-data --disable-backward-weights --verbose";
    std::string bwd         = " --disable-forward --disable-backward-weights --verbose";
    std::string layout      = " --in_layout NHWC --fil_layout NHWC --out_layout NHWC";
    std::string output_int8 = " --output_type int8";
    std::string output_fp32 = " --output_type fp32";
    std::string output_fp16 = " --output_type fp16";
    std::string psd0        = " --pads_strides_dilations 0 0 1 1 1 1";
    std::string psd1        = " --pads_strides_dilations 1 1 1 1 1 1";

    std::vector<std::string> test_cases = {
        // clang-format off
    {precision + fwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_int8 + layout + psd1},
    {precision + fwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_int8 + layout + psd1},
    {precision + fwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_int8 + layout + psd0},
    {precision + fwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_int8 + layout + psd0},

    {precision + fwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_fp32 + layout + psd1},
    {precision + fwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_fp32 + layout + psd1},
    {precision + fwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_fp32 + layout + psd0},
    {precision + fwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_fp32 + layout + psd0},
    {precision + fwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_fp16 + layout + psd1},
    {precision + fwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_fp16 + layout + psd1},
    {precision + fwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_fp16 + layout + psd0},
    {precision + fwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_fp16 + layout + psd0},

    {precision + bwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_fp32 + layout + psd1},
    {precision + bwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_fp32 + layout + psd1},
    {precision + bwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_fp32 + layout + psd0},
    {precision + bwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_fp32 + layout + psd0},
    {precision + bwd + " --input 256 128  28 28 --weights 128  128  3 3" + output_fp16 + layout + psd1},
    {precision + bwd + " --input 128 512  7  7  --weights 512  512  3 3" + output_fp16 + layout + psd1},
    {precision + bwd + " --input 128 64   56 56 --weights 64   64   1 1" + output_fp16 + layout + psd0},
    {precision + bwd + " --input 256 256  56 56 --weights 256  64   1 1" + output_fp16 + layout + psd0}
        // clang-format on
    };

    return test_cases;
}

INSTANTIATE_TEST_SUITE_P(ConvHipIgemmXdlops,
                         ConfigWithInt8,
                         testing::Values(GetTestCases("--int8")));
