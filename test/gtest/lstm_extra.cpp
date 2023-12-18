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

#include "lstm.hpp"
#include "get_handle.hpp"
#include <miopen/env.hpp>
#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_FLOAT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_HALF)

namespace lstm_extra {
using EnvType = std::tuple<std::pair<miopen::env::MIOPEN_TEST_ALL, std::string_view>,
                           std::pair<miopen::env::MIOPEN_TEST_FLOAT, std::string_view>,
                           std::pair<miopen::env::MIOPEN_TEST_HALF, std::string_view>>;

bool Skip(miopenDataType_t prec)
{
    bool flag = !miopen::IsEnabled(ENV(MIOPEN_TEST_ALL));
    switch(prec)
    {
    case miopenFloat: return flag && !miopen::IsEnabled(ENV(MIOPEN_TEST_FLOAT));
    case miopenHalf: return flag && !miopen::IsEnabled(ENV(MIOPEN_TEST_HALF));
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenDouble:
    default: MIOPEN_THROW("Unsupported datatype");
    }
    return true;
}

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

void SetEnv(EnvType env_vars)
{
    std::apply(
        [](const auto&... pairs) {
            (..., (miopen::UpdateEnvVar(pairs.first, std::string_view(pairs.second))));
        },
        env_vars);
}

auto GetTestCases(std::string precision)
{
    const auto env = std::tuple{
        std::pair{ENV(MIOPEN_TEST_ALL), std::string_view("ON")},
        std::pair{ENV(MIOPEN_TEST_FLOAT),
                  precision == "--float" ? std::string_view("ON") : std::string_view("OFF")},
        std::pair{ENV(MIOPEN_TEST_HALF),
                  precision == "--half" ? std::string_view("ON") : std::string_view("OFF")}};

    std::string flags       = "test_lstm --verbose " + precision;
    std::string commonFlags = " --batch-size 32 --seq-len 3 --batch-seq 32 32 32 --vector-len 128 "
                              "--hidden-size 128 --num-layers 1 --in-mode 0 --bias-mode 0";

    return std::vector{
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-hx"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-dhy"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-hx --no-dhy"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-cx"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-hx --no-cx"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-dcy"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-cx --no-dcy"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-hx"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-dhy"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-hx --no-dhy"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-cx"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-hx --no-cx"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-dcy"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-cx --no-dcy"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-hy"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-dhx"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-hy --no-dhx"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-cy"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-hy --no-cy"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-dcx"},
        std::pair{env, flags + commonFlags + " -dir-mode 0 --no-cy --no-dcx"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-hy"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-dhx"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-hy --no-dhx"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-cy"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-hy --no-cy"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-dcx"},
        std::pair{env, flags + commonFlags + " -dir-mode 1 --no-cy --no-dcx"},
        std::pair{
            env,
            flags + commonFlags +
                " -dir-mode 0 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy --no-dcx"},
        std::pair{env,
                  flags + commonFlags +
                      " -dir-mode 1 --no-hx --no-dhy --no-cx --no-dcy --no-hy --no-dhx --no-cy "
                      "--no-dcx"}};
}

using TestCase = decltype(GetTestCases({}))::value_type;

class ConfigWithFloat : public testing::TestWithParam<std::vector<TestCase>>
{
};

class ConfigWithHalf : public testing::TestWithParam<std::vector<TestCase>>
{
};

bool IsTestSupportedForDevice()
{
    using namespace miopen::debug;
    using e_mask = enabled<Gpu::gfx94X, Gpu::gfx103X, Gpu::gfx110X>;
    using d_mask = disabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>;
    return miopen::debug::IsTestSupportedForDevice<d_mask, e_mask>();
}

void Run2dDriver(miopenDataType_t prec)
{
    if(!IsTestSupportedForDevice())
    {
        GTEST_SKIP();
    }
    std::vector<std::pair<EnvType, std::string>> params;
    switch(prec)
    {
    case miopenFloat: params = ConfigWithFloat::GetParam(); break;
    case miopenHalf: params = ConfigWithHalf::GetParam(); break;
    case miopenFloat8:
    case miopenBFloat8:
    case miopenInt8:
    case miopenBFloat16:
    case miopenInt32:
    case miopenDouble:
        FAIL() << "miopenInt8, miopenBFloat16, miopenInt32, "
                  "miopenDouble, miopenFloat8, miopenBFloat8 "
                  "data types not supported by "
                  "lstm_extra test";

    default: params = ConfigWithFloat::GetParam();
    }

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value.second, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });
        SetEnv(test_value.first);
        if(Skip(prec))
        {
            GTEST_SKIP();
        }
        testing::internal::CaptureStderr();
        test_drive<lstm_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

} // namespace lstm_extra
using namespace lstm_extra;

TEST_P(ConfigWithFloat, FloatTest) { Run2dDriver(miopenFloat); };

TEST_P(ConfigWithHalf, HalfTest) { Run2dDriver(miopenHalf); };

INSTANTIATE_TEST_SUITE_P(LstmExtra, ConfigWithFloat, testing::Values(GetTestCases("--float")));

INSTANTIATE_TEST_SUITE_P(LstmExtra, ConfigWithHalf, testing::Values(GetTestCases("--half")));
