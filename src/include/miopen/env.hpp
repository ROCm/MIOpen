/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_ENV_HPP
#define GUARD_MIOPEN_ENV_HPP

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <miopen/errors.hpp>

namespace miopen {

namespace internal {

template <typename T>
struct ParseEnvVal
{
};

template <>
struct ParseEnvVal<bool>
{
    static bool go(const char* vp)
    {
        std::string value_env_str{vp};

        for(auto& c : value_env_str)
        {
            if(std::isalpha(c) != 0)
            {
                c = std::tolower(static_cast<unsigned char>(c));
            }
        }

        if(value_env_str == "disable" || value_env_str == "disabled" || value_env_str == "0" ||
           value_env_str == "no" || value_env_str == "off" || value_env_str == "false")
        {
            return false;
        }
        else if(value_env_str == "enable" || value_env_str == "enabled" || value_env_str == "1" ||
                value_env_str == "yes" || value_env_str == "on" || value_env_str == "true")
        {
            return true;
        }
        else
        {
            MIOPEN_THROW(miopenStatusInvalidValue, "Invalid value for env variable");
        }

        return false; // shouldn't reach here
    }
};

// Supports hexadecimals (with leading "0x"), octals (if prefix is "0") and decimals (default).
// Returns 0 if environment variable is in wrong format (strtoull fails to parse the string).
template <>
struct ParseEnvVal<uint64_t>
{
    static uint64_t go(const char* vp) { return std::strtoull(vp, nullptr, 0); }
};

template <>
struct ParseEnvVal<std::string>
{
    static std::string go(const char* vp) { return std::string{vp}; }
};

template <typename T>
struct EnvVar
{
private:
    T value{};
    bool is_unset = true;

public:
    const T& GetValue() const { return value; }

    bool IsUnset() const { return is_unset; }

    void Unset() const { is_unset = true; }

    void UpdateValue(const T& val)
    {
        is_unset = false;
        value    = val;
    }

    explicit EnvVar(const char* const name, const T& def_val)
    {
        // NOLINTNEXTLINE (concurrency-mt-unsafe)
        const char* vp = std::getenv(name);
        if(vp != nullptr) // a value was provided
        {
            is_unset = false;
            value    = ParseEnvVal<T>::go(vp);
        }
        else // no value provided, use default value
        {
            value = def_val;
        }
    }
};

} // end namespace internal

// static inside function hides the variable and provides
// thread-safety/locking
// Used in global namespace
#define MIOPEN_DECLARE_ENV_VAR(name, type, default_val)                            \
    namespace miopen::env {                                                        \
    struct name                                                                    \
    {                                                                              \
        static_assert(std::is_same_v<name, ::miopen::env::name>,                   \
                      "MIOPEN_DECLARE_ENV* must be used in the global namespace"); \
        using value_type = type;                                                   \
        static miopen::internal::EnvVar<type>& Ref()                               \
        {                                                                          \
            static miopen::internal::EnvVar<type> var{#name, default_val};         \
            return var;                                                            \
        }                                                                          \
    };                                                                             \
    }

#define MIOPEN_DECLARE_ENV_VAR_BOOL(name) MIOPEN_DECLARE_ENV_VAR(name, bool, false)

#define MIOPEN_DECLARE_ENV_VAR_UINT64(name) MIOPEN_DECLARE_ENV_VAR(name, uint64_t, 0)

#define MIOPEN_DECLARE_ENV_VAR_STR(name) MIOPEN_DECLARE_ENV_VAR(name, std::string, "")

#define ENV(name) \
    miopen::env::name {}

/// \todo the following functions should be renamed to either include the word Env
/// or put inside a namespace 'env'. Right now we have a function named Value()
/// that returns env var value as only 64-bit ints

template <class EnvVar>
inline const std::string& GetStringEnv(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, std::string>);
    return EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool IsEnabled(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, bool>);
    return !EnvVar::Ref().IsUnset() && EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool IsDisabled(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, bool>);
    return !EnvVar::Ref().IsUnset() && !EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline uint64_t Value(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, uint64_t>);
    return EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool IsUnset(EnvVar)
{
    return EnvVar::Ref().IsUnset();
}

template <class EnvVar>
void Unset(EnvVar)
{
    EnvVar::Ref().Unset();
}

/// updates the cached value of an environment variable
template <typename EnvVar, typename ValueType>
void UpdateEnvVar(EnvVar, const ValueType& val)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, ValueType>);
    EnvVar::Ref().UpdateValue(val);
}

} // namespace miopen

#endif
