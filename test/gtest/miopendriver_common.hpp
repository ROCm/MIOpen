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
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtest/gtest_env_check_common.hpp>

#include <miopen/process.hpp>
#include <miopen/filesystem.hpp>

#ifdef __linux__
#include <dlfcn.h>
#endif

using ::testing::HasSubstr;
using ::testing::Not;

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_WITH_MIOPENDRIVER)

namespace miopendriver::basearg {
namespace conv {
static const std::string Float    = "conv";
static const std::string Half     = "convfp16";
static const std::string BFloat16 = "convbfp16";
static const std::string Int8     = "convint8";
} // namespace conv

namespace pool {
static const std::string Float = "pool";
static const std::string Half  = "poolfp16";
} // namespace pool

namespace gemm {
static const std::string Float = "gemm";
static const std::string Half  = "gemmfp16";
} // namespace gemm

namespace bn {
static const std::string Float = "bnorm";
static const std::string Half  = "bnormfp16";
} // namespace bn
} // namespace miopendriver::basearg

// Note: Assuming that the MIOpenDriver executable will be beside the testing output location.
static inline miopen::fs::path MIOpenDriverExePath()
{
    static const std::string MIOpenDriverExeName = "MIOpenDriver";

#ifdef __linux__
    miopen::fs::path path = {""};
    Dl_info info;

    if(dladdr(reinterpret_cast<void*>(miopenCreate), &info) != 0)
    {
        path = miopen::fs::canonical(miopen::fs::path{info.dli_fname});
        if(path.empty())
            return path;

        path = path.parent_path();
    }
    return path /= MIOpenDriverExeName;
#else
    return {MIOpenDriverExeName};
#endif
}

static inline void
RunMIOpenDriverTestCommand(const std::vector<std::string>& params,
                           const std::map<std::string_view, std::string_view>& map = {})
{
    for(const auto& param : params)
    {
        int commandResult = 0;
        miopen::Process p{MIOpenDriverExePath()};
        std::vector<char> buffer;

        EXPECT_NO_THROW(commandResult = p.Arguments(param).EnvironmentVariables(map).Capture(buffer).Wait());
        EXPECT_EQ(commandResult, 0)
            << "MIOpenDriver exited with non-zero value when running with arguments: "
            << param;
        std::string result{buffer.begin(), buffer.end()};
        EXPECT_THAT(result, Not(HasSubstr("FAILED")));
    }
}

static inline bool CheckFloatCondition(std::string_view floatArg)
{
    return env::enabled(MIOPEN_TEST_WITH_MIOPENDRIVER) &&
           env::value(MIOPEN_TEST_FLOAT_ARG) == floatArg;
}

static inline bool CheckFloatAndAllCondition(std::string_view floatArg)
{
    return env::enabled(MIOPEN_TEST_ALL) && CheckFloatCondition(floatArg);
}

template <typename disabled_mask, typename enabled_mask>
static inline bool ShouldRunMIOpenDriverTest(const std::string& floatArg, bool skipUnlessAllEnabled)
{
    if(skipUnlessAllEnabled)
    {
        return ShouldRunTestCase<disabled_mask, enabled_mask>(
            [&]() { return CheckFloatAndAllCondition(floatArg); });
    }

    return ShouldRunTestCase<disabled_mask, enabled_mask>(
        [&]() { return CheckFloatCondition(floatArg); });
}
