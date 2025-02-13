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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <miopen/errors.hpp>

namespace miopen::env {

MIOPEN_EXPORT std::optional<std::string> getEnvironmentVariable(std::string_view name);
MIOPEN_EXPORT void setEnvironmentVariable(std::string_view name, std::string_view value);
MIOPEN_EXPORT void clearEnvironmentVariable(std::string_view name);

namespace detail {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T, typename U>
inline constexpr bool is_same_v = std::is_same_v<remove_cvref_t<T>, U>;

template <typename T,
          std::enable_if_t<is_same_v<T, bool> || is_same_v<T, std::string> ||
                               is_same_v<T, std::string_view> || is_same_v<T, unsigned long long>,
                           bool> = true>
struct EnvVar
{
    using value_type = T;

    explicit EnvVar(std::string_view name, T default_value = {}, bool create_if_missing = false)
        : name_{name}, default_value_{std::forward<T>(default_value)}
    {
        auto value = getEnvironmentVariable(name);
        if(!value.has_value())
        {
            if(create_if_missing)
                update(default_value);
        }
        else
        {
            if constexpr(is_same_v<T, unsigned long long>)
            {
                value_ = std::strtoull(value->c_str(), nullptr, 0);
            }
            if constexpr(is_same_v<T, bool>)
            {
                if(value == "0")
                {
                    value_ = false;
                }
                else if(value == "1")
                {
                    value_ = true;
                }
                else
                {
                    std::string value_env_str(value->length(), '\0');
                    std::transform(value->begin(), value->end(), value_env_str.begin(), [](int ch) {
                        return std::tolower(ch);
                    });

                    if(value_env_str == "disable" || value_env_str == "disabled" ||
                       value_env_str == "no" || value_env_str == "off" || value_env_str == "false")
                    {
                        value_ = false;
                    }
                    else if(value_env_str == "enable" || value_env_str == "enabled" ||
                            value_env_str == "yes" || value_env_str == "on" ||
                            value_env_str == "true")
                    {
                        value_ = true;
                    }
                    else
                    {
                        MIOPEN_THROW(miopenStatusInvalidValue,
                                     "Invalid value for env variable (value='" + value_env_str +
                                         "')");
                    }
                }
            }
            if constexpr(is_same_v<T, std::string> || is_same_v<T, std::string_view>)
            {
                value_ = value.value();
            }
        }
    }

    template <typename U>
    U value(std::optional<U> alternate_value = std::nullopt) const
    {
        return static_cast<U>(
            value_.value_or(alternate_value.value_or(static_cast<U>(default_value_))));
    }
    void clear()
    {
        clearEnvironmentVariable(name_);
        value_.reset();
    }

    bool exist() const { return value_.has_value(); }

    template <typename U>
    void update(U value)
    {
        if constexpr(is_same_v<U, std::string> || is_same_v<U, std::string_view>)
        {
            setEnvironmentVariable(name_, value);
        }
        if constexpr(is_same_v<U, bool>)
        {
            setEnvironmentVariable(name_, value ? "1" : "0");
        }
        if constexpr(std::is_integral_v<remove_cvref_t<U>> && !is_same_v<U, bool>)
        {
            setEnvironmentVariable(name_, std::to_string(value));
        }
        value_ = value;
    }
    std::string_view name() const { return name_; }

private:
    std::string_view name_;
    T default_value_;
    std::optional<T> value_{std::nullopt};
};

} // end namespace detail

#define MIOPEN_DECLARE_ENV_VAR(_name, _type, ...)                                  \
    [[maybe_unused]] inline constexpr struct __struct_##_name                      \
    {                                                                              \
        static_assert(std::is_same_v<__struct_##_name, ::__struct_##_name>,        \
                      "MIOPEN_DECLARE_ENV* must be used in the global namespace"); \
        using value_type = _type;                                                  \
        static ::miopen::env::detail::EnvVar<_type>& ref()                         \
        {                                                                          \
            static ::miopen::env::detail::EnvVar<_type> var{#_name, __VA_ARGS__};  \
            return var;                                                            \
        }                                                                          \
        operator ::miopen::env::detail::EnvVar<_type>&() const { return ref(); }   \
        operator bool() const { return ref().exist(); }                            \
        constexpr std::string_view GetName() const { return #_name; }              \
    } _name;

#define MIOPEN_DECLARE_ENV_VAR_BOOL(name, ...) MIOPEN_DECLARE_ENV_VAR(name, bool, __VA_ARGS__)

#define MIOPEN_DECLARE_ENV_VAR_UINT64(name, ...) \
    MIOPEN_DECLARE_ENV_VAR(name, unsigned long long, __VA_ARGS__)

#define MIOPEN_DECLARE_ENV_VAR_STR(name, ...) MIOPEN_DECLARE_ENV_VAR(name, std::string, __VA_ARGS__)

inline bool disabled(const detail::EnvVar<bool>& t) { return !t.template value<bool>(true); }

inline bool enabled(const detail::EnvVar<bool>& t) { return t.template value<bool>(); }

template <typename T, typename U = typename T::value_type>
inline U value(T t)
{
    return std::forward<T>(t).ref().template value<U>();
}

template <typename T, typename U = typename T::value_type>
inline U value_or(T t, U k)
{
    return std::forward<T>(t).ref().template value<U>(std::forward<U>(k));
}

template <typename T>
inline std::string name(T t)
{
    return std::string{std::forward<T>(t).ref().name()};
}

template <typename T>
void clear(T t)
{
    std::forward<T>(t).ref().clear();
}

template <typename T, typename U = typename T::value_type>
inline void update(T t, U k)
{
    std::forward<T>(t).ref().template update<U>(std::forward<U>(k));
}

} // namespace miopen::env

#endif // GUARD_MIOPEN_ENV_HPP
