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
#pragma once

#include <charconv>
#include <string>
#include <string_view>
#include <type_traits>

#include <miopen/env_debug.hpp>

namespace lib_env {

struct LibEnvVar
{
    constexpr LibEnvVar(std::string_view name_in) : name(name_in) {}

    explicit operator bool() const
    {
        return miopen::debug::env::GetEnvVariable(name).has_value();
    }

    std::string_view name;
};

template <class T, std::enable_if_t<std::is_same_v<T, bool> || std::is_same_v<T, std::string> || std::is_same_v<T, std::uint64_t>, bool> = true>
inline T value(const LibEnvVar& env)
{
    const auto value = miopen::debug::env::GetEnvVariable(env.name);
    if(!value)
    {
        MIOPEN_THROW(miopenStatusInternalError);
    }

    if constexpr(std::is_same_v<T, bool>)
    {
        bool bvalue = (value != "0");
        return bvalue;
    }
    else if constexpr(std::is_same_v<T, std::uint64_t>)
    {
        std::uint64_t ullvalue;
        const auto res = std::from_chars(value.value().data(), value.value().data() + value.value().size(), ullvalue);
        if(res.ec == std::errc::invalid_argument || res.ec == std::errc::result_out_of_range)
        {
            MIOPEN_THROW(miopenStatusInvalidValue, "Invalid value for env variable: " + value.value());
        }
        return ullvalue;
    }
    else if constexpr(std::is_same_v<T, std::string>)
    {
        return value.value();
    }
}

inline void update(const LibEnvVar& env, const std::string& value)
{
    miopen::debug::env::UpdateEnvVariable(env.name, value);
}

inline void update(const LibEnvVar& env, std::uint64_t value)
{
    update(env, std::to_string(value));
}

inline void update(const LibEnvVar& env, int value)
{
    update(env, std::to_string(value));
}

inline void update(const LibEnvVar& env, bool value)
{
    update(env, value ? 1 : 0);
}

inline void clear(const LibEnvVar& env)
{
    miopen::debug::env::ClearEnvVariable(env.name);
}

} // namespace lib_env

#define MIOPEN_LIB_ENV_VAR(name) [[maybe_unused]] inline constexpr lib_env::LibEnvVar name(#name);
