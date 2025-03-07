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
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_TUNING_ITERATIONS_MAX) // some uint64 variable
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_FIND_ONLY_SOLVER) // some string variable

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

} // namespace

TEST(CPU_TestGetUnknownEnvVariable_NONE, LibEnvVar)
{
    const std::string_view name = MIOPEN_DEBUG_UNKNOWN_ENVIRONMENT_VARIABLE.name;
    [[gnu::used]] std::optional<std::string> value;
    ASSERT_THROW(value = miopen::debug::env::GetEnvVariable(name), miopen::Exception);
};

TEST(CPU_TestUpdateUnknownEnvVariable_NONE, LibEnvVar)
{
    const std::string_view name = MIOPEN_DEBUG_UNKNOWN_ENVIRONMENT_VARIABLE.name;
    ASSERT_THROW(miopen::debug::env::UpdateEnvVariable(name, "SOME_VALUE"), miopen::Exception);
};

TEST(CPU_TestClearUnknownEnvVariable_NONE, LibEnvVar)
{
    const std::string_view name = MIOPEN_DEBUG_UNKNOWN_ENVIRONMENT_VARIABLE.name;
    ASSERT_THROW(miopen::debug::env::ClearEnvVariable(name), miopen::Exception);
};

TEST(CPU_TestEnvVariableRestore_NONE, LibEnvVar)
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

TEST(CPU_TestEnvVariableBool_NONE, LibEnvVar)
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
};

TEST(CPU_TestEnvVariableUInt64_NONE, LibEnvVar)
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
};

TEST(CPU_TestEnvVariableStr_NONE, LibEnvVar)
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
};

TEST(CPU_TestEnvVariableWrapper_NONE, LibEnvVar)
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
};
