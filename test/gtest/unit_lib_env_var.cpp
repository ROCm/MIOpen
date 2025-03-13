/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#include <miopen/errors.hpp>
#include "../lib_env_var.hpp"

namespace {

MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_UNKNOWN_ENVIRONMENT_VARIABLE)

// If these env variables are removed from the library, they need to be replaced with some others.
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS) // some bool variable
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX)     // some uint64 variable
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_FIND_ONLY_SOLVER)          // some string variable

class EnvVarRestorer
{
public:
    EnvVarRestorer(std::string_view name_in) : name(name_in)
    {
        prev = miopen::debug::env::GetEnvVariable(name);
    }

    ~EnvVarRestorer()
    {
        if(prev.has_value())
            miopen::debug::env::UpdateEnvVariable(name, prev.value());
        else
            miopen::debug::env::ClearEnvVariable(name);
    }

private:
    std::string_view name;
    std::optional<std::string> prev;
};

struct TestParams
{
    friend std::ostream& operator<<(std::ostream& os, const TestParams& tp)
    {
        os << "none";
        return os;
    }
};

const auto& GetTestParams()
{
    static const auto params = TestParams{};
    return params;
}

class UnitTestLibEnvVarGetUnknownVariable : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const std::string_view name = MIOPEN_DEBUG_UNKNOWN_ENVIRONMENT_VARIABLE.name;
        [[gnu::used]] std::optional<std::string> value;
        ASSERT_THROW(value = miopen::debug::env::GetEnvVariable(name), miopen::Exception);
    }
};

class UnitTestLibEnvVarUpdateUnknownVariable : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const std::string_view name = MIOPEN_DEBUG_UNKNOWN_ENVIRONMENT_VARIABLE.name;
        ASSERT_THROW(miopen::debug::env::UpdateEnvVariable(name, "SOME_VALUE"), miopen::Exception);
    }
};

class UnitTestLibEnvVarClearUnknownVariable : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const std::string_view name = MIOPEN_DEBUG_UNKNOWN_ENVIRONMENT_VARIABLE.name;
        ASSERT_THROW(miopen::debug::env::ClearEnvVariable(name), miopen::Exception);
    }
};

class UnitTestLibEnvVarRestore : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const std::string_view name = MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS.name;

        const auto old_v = miopen::debug::env::GetEnvVariable(name);
        if(!old_v.has_value())
        {
            {
                EnvVarRestorer restorer(name);
                miopen::debug::env::UpdateEnvVariable(name, "0");
            }
            const auto new_v = miopen::debug::env::GetEnvVariable(name);
            ASSERT_TRUE(!new_v.has_value());
        }
        else
        {
            {
                EnvVarRestorer restorer(name);
                miopen::debug::env::ClearEnvVariable(name);
            }
            auto new_v = miopen::debug::env::GetEnvVariable(name);
            ASSERT_TRUE(new_v.has_value());
            ASSERT_TRUE(new_v.value() == old_v.value());

            {
                EnvVarRestorer restorer(name);
                miopen::debug::env::UpdateEnvVariable(name, "1");
            }
            new_v = miopen::debug::env::GetEnvVariable(name);
            ASSERT_TRUE(new_v.has_value());
            ASSERT_TRUE(new_v.value() == old_v.value());

            {
                EnvVarRestorer restorer(name);
                miopen::debug::env::UpdateEnvVariable(name, "0");
            }
            new_v = miopen::debug::env::GetEnvVariable(name);
            ASSERT_TRUE(new_v.has_value());
            ASSERT_TRUE(new_v.value() == old_v.value());
        }
    }
};

class UnitTestLibEnvVarBool : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const std::string_view name = MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS.name;
        EnvVarRestorer restorer(name);

        // Set 0
        miopen::debug::env::ClearEnvVariable(name);
        miopen::debug::env::UpdateEnvVariable(name, "0");
        ASSERT_EQ(miopen::debug::env::GetEnvVariable(name), "0");

        // Set 1
        miopen::debug::env::ClearEnvVariable(name);
        miopen::debug::env::UpdateEnvVariable(name, "1");
        ASSERT_EQ(miopen::debug::env::GetEnvVariable(name), "1");

        // Set 100 --> 1
        miopen::debug::env::ClearEnvVariable(name);
        miopen::debug::env::UpdateEnvVariable(name, "100");
        ASSERT_EQ(miopen::debug::env::GetEnvVariable(name), "1");
    }
};

class UnitTestLibEnvVarUInt64 : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const std::string_view name = MIOPEN_DEBUG_TUNING_ITERATIONS_MAX.name;
        EnvVarRestorer restorer(name);

        // Set 0
        miopen::debug::env::ClearEnvVariable(name);
        miopen::debug::env::UpdateEnvVariable(name, "0");
        ASSERT_EQ(miopen::debug::env::GetEnvVariable(name), "0");

        // Set 1
        miopen::debug::env::ClearEnvVariable(name);
        miopen::debug::env::UpdateEnvVariable(name, "1");
        ASSERT_EQ(miopen::debug::env::GetEnvVariable(name), "1");

        // Set 18446744073709551615
        miopen::debug::env::ClearEnvVariable(name);
        miopen::debug::env::UpdateEnvVariable(name, "18446744073709551615");
        ASSERT_EQ(miopen::debug::env::GetEnvVariable(name), "18446744073709551615");
    }
};

class UnitTestLibEnvVarString : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const std::string_view name = MIOPEN_DEBUG_FIND_ONLY_SOLVER.name;
        EnvVarRestorer restorer(name);

        // Set 0
        miopen::debug::env::ClearEnvVariable(name);
        miopen::debug::env::UpdateEnvVariable(name, "0");
        ASSERT_EQ(miopen::debug::env::GetEnvVariable(name), "0");

        // Set asdfghjkl
        miopen::debug::env::ClearEnvVariable(name);
        miopen::debug::env::UpdateEnvVariable(name, "asdfghjkl");
        ASSERT_EQ(miopen::debug::env::GetEnvVariable(name), "asdfghjkl");

        // Set qwertyuiop
        miopen::debug::env::ClearEnvVariable(name);
        miopen::debug::env::UpdateEnvVariable(name, "qwertyuiop");
        ASSERT_EQ(miopen::debug::env::GetEnvVariable(name), "qwertyuiop");
    }
};

class UnitTestLibEnvVarWrapper : public ::testing::TestWithParam<TestParams>
{
public:
    void RunTest()
    {
        const auto var = MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS;
        EnvVarRestorer restorer(var.name);

        // Set false
        lib_env::clear(var);
        lib_env::update(var, false);
        ASSERT_EQ(lib_env::value<bool>(var), false);
        ASSERT_EQ(lib_env::value<uint64_t>(var), 0);
        ASSERT_EQ(lib_env::value<std::string>(var), std::string("0"));

        // Set true
        lib_env::clear(var);
        lib_env::update(var, true);
        ASSERT_EQ(lib_env::value<bool>(var), true);
        ASSERT_EQ(lib_env::value<uint64_t>(var), 1);
        ASSERT_EQ(lib_env::value<std::string>(var), std::string("1"));

        // Set "0"
        lib_env::clear(var);
        lib_env::update(var, "0");
        ASSERT_EQ(lib_env::value<bool>(var), false);
        ASSERT_EQ(lib_env::value<uint64_t>(var), 0);
        ASSERT_EQ(lib_env::value<std::string>(var), std::string("0"));

        // Set "1"
        lib_env::clear(var);
        lib_env::update(var, "1");
        ASSERT_EQ(lib_env::value<bool>(var), true);
        ASSERT_EQ(lib_env::value<uint64_t>(var), 1);
        ASSERT_EQ(lib_env::value<std::string>(var), std::string("1"));

        // Set std::string("0")
        lib_env::clear(var);
        lib_env::update(var, std::string("0"));
        ASSERT_EQ(lib_env::value<bool>(var), false);
        ASSERT_EQ(lib_env::value<uint64_t>(var), 0);
        ASSERT_EQ(lib_env::value<std::string>(var), std::string("0"));

        // Set std::string("1")
        lib_env::clear(var);
        lib_env::update(var, std::string("1"));
        ASSERT_EQ(lib_env::value<bool>(var), true);
        ASSERT_EQ(lib_env::value<uint64_t>(var), 1);
        ASSERT_EQ(lib_env::value<std::string>(var), std::string("1"));

        // Set std::string_view("0")
        lib_env::clear(var);
        lib_env::update(var, std::string_view("0"));
        ASSERT_EQ(lib_env::value<bool>(var), false);
        ASSERT_EQ(lib_env::value<uint64_t>(var), 0);
        ASSERT_EQ(lib_env::value<std::string>(var), std::string("0"));

        // Set std::string_view("1")
        lib_env::clear(var);
        lib_env::update(var, std::string_view("1"));
        ASSERT_EQ(lib_env::value<bool>(var), true);
        ASSERT_EQ(lib_env::value<uint64_t>(var), 1);
        ASSERT_EQ(lib_env::value<std::string>(var), std::string("1"));
    }
};

} // namespace

using CPU_UnitTestLibEnvVarGetUnknownVariable_NONE    = UnitTestLibEnvVarGetUnknownVariable;
using CPU_UnitTestLibEnvVarUpdateUnknownVariable_NONE = UnitTestLibEnvVarUpdateUnknownVariable;
using CPU_UnitTestLibEnvVarClearUnknownVariable_NONE  = UnitTestLibEnvVarClearUnknownVariable;
using CPU_UnitTestLibEnvVarRestore_NONE               = UnitTestLibEnvVarRestore;
using CPU_UnitTestLibEnvVarBool_NONE                  = UnitTestLibEnvVarBool;
using CPU_UnitTestLibEnvVarUInt64_NONE                = UnitTestLibEnvVarUInt64;
using CPU_UnitTestLibEnvVarString_NONE                = UnitTestLibEnvVarString;
using CPU_UnitTestLibEnvVarWrapper_NONE               = UnitTestLibEnvVarWrapper;

TEST_P(CPU_UnitTestLibEnvVarGetUnknownVariable_NONE, LibEnvVar) { this->RunTest(); };
TEST_P(CPU_UnitTestLibEnvVarUpdateUnknownVariable_NONE, LibEnvVar) { this->RunTest(); };
TEST_P(CPU_UnitTestLibEnvVarClearUnknownVariable_NONE, LibEnvVar) { this->RunTest(); };
TEST_P(CPU_UnitTestLibEnvVarRestore_NONE, LibEnvVar) { this->RunTest(); }
TEST_P(CPU_UnitTestLibEnvVarBool_NONE, LibEnvVar) { this->RunTest(); };
TEST_P(CPU_UnitTestLibEnvVarUInt64_NONE, LibEnvVar) { this->RunTest(); };
TEST_P(CPU_UnitTestLibEnvVarString_NONE, LibEnvVar) { this->RunTest(); };
TEST_P(CPU_UnitTestLibEnvVarWrapper_NONE, LibEnvVar) { this->RunTest(); };

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_UnitTestLibEnvVarGetUnknownVariable_NONE,
                         testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_UnitTestLibEnvVarUpdateUnknownVariable_NONE,
                         testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_UnitTestLibEnvVarClearUnknownVariable_NONE,
                         testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full, CPU_UnitTestLibEnvVarRestore_NONE, testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full, CPU_UnitTestLibEnvVarBool_NONE, testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full, CPU_UnitTestLibEnvVarUInt64_NONE, testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full, CPU_UnitTestLibEnvVarString_NONE, testing::Values(GetTestParams()));

INSTANTIATE_TEST_SUITE_P(Full, CPU_UnitTestLibEnvVarWrapper_NONE, testing::Values(GetTestParams()));
