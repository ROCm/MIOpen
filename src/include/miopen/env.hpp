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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <miopen/errors.hpp>

namespace miopen {

std::optional<std::string> getEnvironmentVariable(std::string_view name);
void setEnvironmentVariable(std::string_view name, std::string_view value);
void unsetEnvironmentVariable(std::string_view name);

namespace internal {

template <typename T>
struct EnvVar
{
    static_assert(false, "data type not supported");
};

template <>
struct EnvVar<unsigned long long>
{
    explicit EnvVar(std::string_view name, unsigned long long default_value = 0)
        : name_{name}, default_value_{default_value}
    {
    }

    std::optional<unsigned long long> getValue(bool use_default = true) const
    {
        auto value = getEnvironmentVariable(name_);
        if(value.has_value())
        {
            return std::strtoull(value->c_str(), nullptr, 0);
        }
        return use_default ? default_value_ : std::nullopt;
    }

    bool isSet() const
    {
        return getEnvironmentVariable(name_).has_value();
    }

    void unset() const
    {
        unsetEnvironmentVariable(name_);
    }

    void updateValue(unsigned long long value) const
    {
        setEnvironmentVariable(name_, std::to_string(value));
    }

private:
    std::string_view name_;
    std::optional<unsigned long long> default_value_;
};

template <>
struct EnvVar<bool>
{
    explicit EnvVar(std::string_view name, bool default_value = false)
        : name_{name}, default_value_{default_value}
    {
    }

    template <bool use_default = true>
    std::optional<bool> getValue() const
    {
        auto value = getEnvironmentVariable(name_);
        if(!value.has_value())
        {
            return use_default ? default_value_ : std::nullopt;
        }

        if(value == "0")
            return false;

        if(value == "1")
            return true;

        std::string value_env_str{};
        std::transform(value->begin(), value->end(), value_env_str.begin(),
                       [](int ch) { return std::tolower(ch); });

        if(value_env_str == "disable" || value_env_str == "disabled" ||
            value_env_str == "no" || value_env_str == "off" || value_env_str == "false")
        {
            return false;
        }

        if(value_env_str == "enable" || value_env_str == "enabled" ||
           value_env_str == "yes" || value_env_str == "on" || value_env_str == "true")
        {
            return true;
        }

        MIOPEN_THROW(miopenStatusInvalidValue, "Invalid value for env variable");
    }

    bool isSet() const
    {
        return getEnvironmentVariable(name_).has_value();
    }

    void unset() const
    {
        unsetEnvironmentVariable(name_);
    }

    void updateValue(bool value) const
    {
        setEnvironmentVariable(name_, value ? "1" : "0");
    }

private:
    std::string_view name_;
    std::optional<bool> default_value_;
};

template <>
struct EnvVar<std::string>
{
    explicit EnvVar(std::string_view name, std::string_view default_value = "")
        : name_{name}, default_value_{default_value}
    {
    }

    std::optional<std::string> getValue(bool use_default = true) const
    {
        auto value = getEnvironmentVariable(name_);
        return value ? value : use_default ? default_value_ : std::nullopt;
    }

    bool isSet() const
    {
        return getEnvironmentVariable(name_).has_value();
    }

    void unset() const
    {
        unsetEnvironmentVariable(name_);
    }

    void updateValue(std::string_view value) const
    {
        setEnvironmentVariable(name_, value);
    }
private:
    std::string_view name_;
    std::optional<std::string> default_value_;
};

} // end namespace internal

// static inside function hides the variable and provides
// thread-safety/locking
// Used in global namespace
#define MIOPEN_DECLARE_ENV_VAR(name, type, ...)                                    \
    namespace miopen::env {                                                        \
    struct name                                                                    \
    {                                                                              \
        static_assert(std::is_same_v<name, ::miopen::env::name>,                   \
                      "MIOPEN_DECLARE_ENV* must be used in the global namespace"); \
        using value_type = type;                                                   \
        static const miopen::internal::EnvVar<type>& Ref()                         \
        {                                                                          \
            static miopen::internal::EnvVar<type> var{#name, __VA_ARGS__};         \
            return var;                                                            \
        }                                                                          \
    };                                                                             \
    }

#define MIOPEN_DECLARE_ENV_VAR_BOOL(name, ...) \
    MIOPEN_DECLARE_ENV_VAR(name, bool, __VA_ARGS__)

#define MIOPEN_DECLARE_ENV_VAR_UINT64(name, ...) \
    MIOPEN_DECLARE_ENV_VAR(name, unsigned long long, __VA_ARGS__)

#define MIOPEN_DECLARE_ENV_VAR_STR(name, ...) \
    MIOPEN_DECLARE_ENV_VAR(name, std::string, __VA_ARGS__)

#define ENV(name) \
    miopen::env::name {}

/// \todo the following functions should be renamed to either include the word Env
/// or put inside a namespace 'env'. Right now we have a function named Value()
/// that returns env var value as only 64-bit ints

template <class EnvVar>
inline std::string GetStringEnv(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, std::string>);
    return *EnvVar::Ref().getValue();
}

template <class EnvVar>
inline bool IsEnabled(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, bool>);
    return EnvVar::Ref().template getValue<false>().value_or(false);
}

template <class EnvVar>
inline bool IsDisabled(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, bool>);
    // The original code had a bug here, and unfortunately, the bug is being utilized in the rest
    // of the code. Initially, it was "!EnvVar::Ref().IsUnset() && !EnvVar::Ref().GetValue()".
    // The sentence means the variable is ENABLED even if it is not set in the environment block!!!
    // Hence, 'false' is here and not 'true' to accommodate this bug, so algorithms are not broken.
    return EnvVar::Ref().template getValue<false>().value_or(false);
}

template <class EnvVar>
inline unsigned long long Value(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, unsigned long long>);
    return *EnvVar::Ref().getValue();
}

template <class EnvVar>
inline bool IsSet(EnvVar)
{
    return EnvVar::Ref().isSet();
}

template <class EnvVar>
void Unset(EnvVar)
{
    EnvVar::Ref().unset();
}

template <typename T>
using _rm_cvref = std::remove_cv_t<std::remove_reference_t<T>>;

/// updates the value of an environment variable
template <typename EnvVar, typename ValueType>
std::enable_if_t<
    std::is_integral_v<_rm_cvref<typename EnvVar::value_type>> && std::is_integral_v<ValueType>>
UpdateEnvVar(EnvVar, const ValueType& value)
{
    EnvVar::Ref().updateValue(value);
}

template <typename EnvVar>
void UpdateEnvVar(EnvVar, std::string_view value)
{
    EnvVar::Ref().updateValue(value);
}

} // namespace miopen

#endif
