#include <tuple>

#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/env.hpp>
#include "conv_2d.hpp"
#include "get_handle.hpp"


using TestCase = std::tuple<std::vector<std::string>, std::string>;


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

MIOPEN_DECLARE_ENV_VAR(MIOPEN_TEST_FLOAT_ARG)




class ConfigWithHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithBF16 : public testing::TestWithParam<std::vector<TestCase>>
{
};

void Run2dDriver(miopenDataType_t prec)
{
    std::vector<TestCase> params;

    switch (prec)
    {
        case miopenHalf :
            /* code */
            params = ConfigWithHalf::GetParam();
            break;
        
        case miopenBFloat16:
            params = ConfigWithBF16::GetParam();
            break;

        default:params = ConfigWithHalf::GetParam();


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
        std::cout << capture;

    }

}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// How I should write devName for 
// https://github.com/ROCmSoftwarePlatform/MIOpen/blob/develop/test/CMakeLists.txt#L845


bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    std::string devName = handle.GetDeviceName();
    if(devName == "gfx94X" || devName == "gfx103x")
        return true;
    else
        return false;
}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// How I should write Envvar Values 
// https://github.com/ROCmSoftwarePlatform/MIOpen/blob/develop/test/CMakeLists.txt#L845


TEST_P(ConfigWithBF16, BF16Test)
{
#if MIOPEN_BACKEND_OPENCL

    GTEST_SKIP() << "MIOPEN_BACKEND_HIP needed for this test";

#else // MIOPEN_BACKEND_HIP, OCL_DISABLED
    const auto& handle = get_handle();
    
    if(IsTestSupportedForDevice(handle) &&
       miopen::IsEnvvarValueEnabled("MIOPEN_TEST_COMPOSABLEKERNEL") &&
       miopen::IsEnvvarValueEnabled("MIOPEN_TEST_ALL") && IsTestRunWith("--bf16"))
    {
        Run2dDriver(miopenBFloat16);
    }
    else
    {
        GTEST_SKIP();
    }
#endif
};

TEST_P(ConfigWithHalf, HalfTest)
{
#if MIOPEN_BACKEND_OPENCL

    GTEST_SKIP() << "MIOPEN_BACKEND_HIP needed for this test";

#else // MIOPEN_BACKEND_HIP, OCL_DISABLED
    const auto& handle = get_handle();
    
    if(IsTestSupportedForDevice(handle) &&
       miopen::IsEnvvarValueEnabled("MIOPEN_TEST_COMPOSABLEKERNEL") &&
       miopen::IsEnvvarValueEnabled("MIOPEN_TEST_ALL") && IsTestRunWith("--half"))
    {
        Run2dDriver(miopenHalf);
    }
    else
    {
        GTEST_SKIP();
    }
#endif
};

std::vector<TestCase> GetTestCases(const std::string& precision){




    
}





INSTANTIATE_TEST_SUITE_P(ConvIgemm,
                             ConfigWithBF16, 
                             testing::Values(GetTestCases("--bf16")));

INSTANTIATE_TEST_SUITE_P(ConvIgemm,
                             ConfigWithHalf, 
                             testing::Values(GetTestCases("--half")));