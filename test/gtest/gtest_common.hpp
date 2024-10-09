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

#pragma once

#include <algorithm>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <miopen/env.hpp>
#include <tuple>
#include <sstream>
#include <string>
#include <vector>

#include "../driver.hpp"

inline void default_check(const std::string& err) { std::cout << err; }

inline void tuning_check(const std::string& err)
{
    // TEST_TUNING - the test should fail if output contains "Error" or "failed".
    EXPECT_FALSE(err.find("Error") != std::string::npos || err.find("failed") != std::string::npos);
    default_check(err);
}

inline void db_check(const std::string& err)
{
    EXPECT_FALSE(err.find("Perf Db: record not found") != std::string::npos);
    default_check(err);
};

enum class Gpu : int
{
    Default = 0, // \todo remove
    None    = 0,
    gfx900  = 1 << 0,
    gfx906  = 1 << 1,
    gfx908  = 1 << 2,
    gfx90A  = 1 << 3,
    gfx94X  = 1 << 4,
    gfx103X = 1 << 5,
    gfx110X = 1 << 6,
    gfx120X = 1 << 7,
    gfxLast = Gpu::gfx120X, // \note Change the value when adding a new device
    All     = -1
};

inline Gpu operator|(Gpu lhs, Gpu rhs)
{
    using T = std::underlying_type_t<Gpu>;
    return static_cast<Gpu>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline Gpu operator&(Gpu lhs, Gpu rhs)
{
    using T = std::underlying_type_t<Gpu>;
    return static_cast<Gpu>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

template <Gpu... bits>
struct enabled
{
    using T                        = std::underlying_type_t<Gpu>;
    static constexpr T val         = (static_cast<T>(bits) | ...);
    static constexpr bool enabling = true;
};

template <Gpu... bits>
struct disabled
{
    using T                        = std::underlying_type_t<Gpu>;
    static constexpr T val         = ~((static_cast<T>(bits) | ...));
    static constexpr bool enabling = false;
};

struct DevDescription
{
    std::string_view name;
    unsigned cu_cnt; // CU for gfx9, WGP for gfx10, 11, ...

    friend std::ostream& operator<<(std::ostream& os, const DevDescription& dd);
};

class MockHandle final : public miopen::Handle
{
public:
    MockHandle(const DevDescription& dev_description);

    // Add additional methods here if needed
    std::string GetDeviceName() const override;
    std::size_t GetMaxComputeUnits() const override;
    std::size_t GetMaxMemoryAllocSize() override;
    bool CooperativeLaunchSupported() const override;

private:
    DevDescription dev_descr;
};

Gpu GetDevGpuType();
const std::multimap<Gpu, DevDescription>& GetAllKnownDevices();
bool IsTestSupportedByDevice(Gpu supported_devs);

template <typename disabled_mask, typename enabled_mask>
bool IsTestSupportedForDevMask()
{
    static_assert((~disabled_mask::val & enabled_mask::val) == 0,
                  "Enabled and Disabled GPUs are overlapped");
    static_assert(disabled_mask::enabling == false,
                  "Wrong disabled mask, probably it has to be switched with enabled_mask");
    static_assert(enabled_mask::enabling == true,
                  "Wrong enabled mask, probably it has to be switched with disabled_mask");

    constexpr int def_val = enabled<Gpu::gfx900, Gpu::gfx906, Gpu::gfx908, Gpu::gfx90A>::val;
    constexpr int mask    = (def_val & disabled_mask::val) | enabled_mask::val;

    return IsTestSupportedByDevice(static_cast<Gpu>(mask));
}

template <typename Parameters>
struct FloatTestCase : public testing::TestWithParam<Parameters>
{
    static constexpr std::string_view fp_args{"--float"};
};

template <typename Parameters>
struct HalfTestCase : public testing::TestWithParam<Parameters>
{
    static constexpr std::string_view fp_args{"--half"};
};

template <typename Parameters>
struct Bf16TestCase : public testing::TestWithParam<Parameters>
{
    static constexpr std::string_view fp_args{"--bfloat16"};
};

template <typename Parameters>
struct Int8TestCase : public testing::TestWithParam<Parameters>
{
    static constexpr std::string_view fp_args{"--int8"};
};

template <typename Case>
std::vector<std::string> get_args(const Case& param)
{
    const auto& [env_tuple, cmd] = param;
    std::apply([](const auto&... env) { (env::update(env.first, env.second), ...); }, env_tuple);

    std::stringstream ss(cmd);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;

    return {begin, end};
}

template <template <class...> class Driver, typename TestCase, typename Check>
void invoke_with_params(Check&& check)
{
    for(const auto& test_value : TestCase::GetParam())
    {
        std::vector<std::string> tokens = get_args(test_value);
        std::vector<const char*> ptrs;
        ptrs.reserve(tokens.size() + 1);
        ptrs.emplace_back(TestCase::fp_args.data());

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<Driver>(ptrs.size(), ptrs.data(), "unnamed");
        check(testing::internal::GetCapturedStderr());
    }
}

/// repalcement of gtest's FAIL() because FAIL() returns void and messes up the
/// return type deduction
#define MIOPEN_FRIENDLY_FAIL(MSG)   \
    do                              \
    {                               \
        [&]() { FAIL() << MSG; }(); \
    } while(false);
/// The types for env variables must be redefined, but
/// do not mess up with the types - those variables are decalred in the library
/// and if wrong type (STR|BOOl|UINT64) have been specified they won't be updated. Silently.
/// There will be no compiler warnings or runtime errors.
/// TODO: move ALL the env variables to the single header int the library to avoid such problems
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_FIND_ONLY_SOLVER)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_MODE)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_WINOGRAD)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_FFT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GEMM)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_LOG_LEVEL)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DRIVER_USE_GPU_REFERENCE)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_FIND_ENFORCE)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V1R1_XDLOPS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_V4R1)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_BWD_V4R1_XDLOPS)
