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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
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
#ifndef _WIN32
    miopen::fs::path path{};
    Dl_info info;
    if(dladdr(reinterpret_cast<void*>(miopenCreate), &info) != 0)
        path = miopen::fs::canonical(miopen::fs::path{info.dli_fname});
    if(path.has_parent_path())
        path = path.parent_path();
    return path / "MIOpenDriver";
#else
    HMODULE module = nullptr;
    if(GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(miopenCreate),
                         &module) == 0)
    {
        throw std::runtime_error{"unable to obtain module handle (" +
                                 std::to_string(GetLastError()) + ")"};
    }
    constexpr std::size_t PATH_MAX = 32767;
    TCHAR buffer[PATH_MAX];
    if(GetModuleFileName(module, buffer, sizeof(buffer)) == 0)
    {
        throw std::runtime_error{"unable to read module file path (" +
                                 std::to_string(GetLastError()) + ")"};
    }
    if(GetLastError() == ERROR_INSUFFICIENT_BUFFER)
    {
        throw std::runtime_error{"buffer too small, (" + std::to_string(PATH_MAX) +
                                 ") to hold the path"};
    }
    miopen::fs::path path{buffer};
    if(path.has_parent_path())
        path = path.parent_path();
    return path / "MIOpenDriver.exe";
#endif
}

static inline void
RunMIOpenDriverTestCommand(const std::vector<std::string>& params,
                           const std::map<std::string_view, std::string_view>& map = {})
{
    for(const auto& param : params)
    {
        int commandResult = 0;
        miopen::Process p{MIOpenDriverExePath(), param};
        std::vector<char> buffer;

        EXPECT_NO_THROW(commandResult = p.EnvironmentVariables(map).Read(buffer).Wait());
        EXPECT_EQ(commandResult, 0)
            << "MIOpenDriver exited with non-zero value when running with arguments: " << param;
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
