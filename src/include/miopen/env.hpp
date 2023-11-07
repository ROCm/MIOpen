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

namespace miopen {

namespace internal {

/* NOTES AND GOTCHAS (TODO: Remove before committing)
 * 1. All env variables that are used with IsDisabled() should be declared with
 *    default value 'true' using: MIOPEN_DECLARE_ENV_VAR(name, bool, true)
 * 2. 
 */

template <typename T>
struct ParseEnvVal{};

template <>
struct ParseEnvVal<bool> {
  bool go(const char* vp) {
    std::string value_env_str{vp};

    for(auto& c : value_env_str)
    {
        if(std::isalpha(c) != 0)
        {
            c = std::tolower(static_cast<unsigned char>(c));
        }
    }

    if (std::strcmp(value_env_str.c_str(), "disable") == 0 ||
        std::strcmp(value_env_str.c_str(), "disabled") == 0 ||
        std::strcmp(value_env_str.c_str(), "0") == 0 ||
        std::strcmp(value_env_str.c_str(), "no") == 0 ||
        std::strcmp(value_env_str.c_str(), "off") == 0 ||
        std::strcmp(value_env_str.c_str(), "false") == 0)
    {
      return false;
    }
    else if (std::strcmp(value_env_str.c_str(), "enable") == 0 ||
             std::strcmp(value_env_str.c_str(), "enabled") == 0 ||
             std::strcmp(value_env_str.c_str(), "1") == 0 ||
             std::strcmp(value_env_str.c_str(), "yes") == 0 ||
             std::strcmp(value_env_str.c_str(), "on") == 0 ||
             std::strcmp(value_env_str.c_str(), "true") == 0)
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

template <>
struct ParseEnvVal<uint64_t> {
  uint64_t go(const char* vp) {
    return std::strtoull(vp, nullptr, 0);
  }
};

template <>
struct ParseEnvVal<std::string> {
  std::string go(const char* vp) {
    return std::string{vp};
  }
};

template <typename T>
struct EnvVar {

  const T& GetValue() const {
    return value;
  }

  void UpdateValue(const T& val) {
    value = val;
  }

  explicit EnvVar(const char* const name, const T& def_val) {
    const char* vp = std::getenv(name);
    if (vp) // a value was provided
    {
      if constexpr (std::is_same_v<T, bool>) 
      {
        value = ParseEnvVal<bool>::go(vp);
      }
      else if constexpr (std::is_same_v<T, uint64_t>) {
        value = ParseEnvVal<uint64_t>::go(vp);
      } else if constexpr (std::is_same_v<T, std::string>) {
        value = ParseEnvVal<std::string>::go(vp);
      } else {
        value = ParseEnvVal<T>::go(vp); // should cause compile error
      }
    }
    else  // no value provided, use default value
    {
      value = def_val;
    }

  private: 
    T value{};
};


}// end namespace internal


#if 1 
// static inside function hides the variable and provides
// thread-safety/locking 
#define MIOPEN_DECLARE_ENV_VAR(name, type, default_val) \
  struct name { \
    using value_type = type; \
    static internal::EnvVar<type>& Ref() { \
      static internal::EnvVar<type> var{#name, default_val}; \
      return var;\
    }\
  }; 

#else 
/// \todo Rework: Case-insensitive string compare, ODR, (?) move to .cpp

// Declare a cached environment variable
#define MIOPEN_DECLARE_ENV_VAR(x)                 \
    struct x                                      \
    {                                             \
        static const char* value() { return #x; } \
    };

/*
 * Returns false if a feature-controlling environment variable is defined
 * and set to something which disables a feature.
 */
inline bool IsEnvvarValueDisabled(const char* name)
{
    // NOLINTNEXTLINE (concurrency-mt-unsafe)
    const auto value_env_p = std::getenv(name);
    if(value_env_p == nullptr)
        return false;
    else
    {
        std::string value_env_str = value_env_p;
        for(auto& c : value_env_str)
        {
            if(std::isalpha(c) != 0)
            {
                c = std::tolower(static_cast<unsigned char>(c));
            }
        }
        return (std::strcmp(value_env_str.c_str(), "disable") == 0 ||
                std::strcmp(value_env_str.c_str(), "disabled") == 0 ||
                std::strcmp(value_env_str.c_str(), "0") == 0 ||
                std::strcmp(value_env_str.c_str(), "no") == 0 ||
                std::strcmp(value_env_str.c_str(), "off") == 0 ||
                std::strcmp(value_env_str.c_str(), "false") == 0);
    }
}

inline bool IsEnvvarValueEnabled(const char* name)
{
    // NOLINTNEXTLINE (concurrency-mt-unsafe)
    const auto value_env_p = std::getenv(name);
    if(value_env_p == nullptr)
        return false;
    else
    {
        std::string value_env_str = value_env_p;
        for(auto& c : value_env_str)
        {
            if(std::isalpha(c) != 0)
            {
                c = std::tolower(static_cast<unsigned char>(c));
            }
        }
        return (std::strcmp(value_env_str.c_str(), "enable") == 0 ||
                std::strcmp(value_env_str.c_str(), "enabled") == 0 ||
                std::strcmp(value_env_str.c_str(), "1") == 0 ||
                std::strcmp(value_env_str.c_str(), "yes") == 0 ||
                std::strcmp(value_env_str.c_str(), "on") == 0 ||
                std::strcmp(value_env_str.c_str(), "true") == 0);
    }
}

// Return 0 if env is enabled else convert environment var to an int.
// Supports hexadecimal with leading 0x or decimal
inline uint64_t EnvvarValue(const char* name, uint64_t fallback = 0)
{
    // NOLINTNEXTLINE (concurrency-mt-unsafe)
    const auto value_env_p = std::getenv(name);
    if(value_env_p == nullptr)
    {
        return fallback;
    }
    else
    {
        return strtoull(value_env_p, nullptr, 0);
    }
}

inline std::vector<std::string> GetEnv(const char* name)
{
    // NOLINTNEXTLINE (concurrency-mt-unsafe)
    const auto p = std::getenv(name);
    if(p == nullptr)
        return {};
    else
        return {{p}};
}
#endif

/// \todo the following functions should be renamed to either include the word Env
/// or put inside a namespace 'env'. Right now we have a function named Value()
/// that returns env var value as only 64-bit ints

template <class EnvVar>
inline std::string GetStringEnv(EnvVar)
{
    static_assert(std::is_same_v<EnvVar::value_type, std::string>);
    return EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool IsEnabled(EnvVar)
{
    static_assert(std::is_same_v<EnvVar::value_type, bool>);
    return EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool IsDisabled(EnvVar)
{
    static_assert(std::is_same_v<EnvVar::value_type, bool>);
    return !EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline uint64_t Value(EnvVar)
{
    static_assert(std::is_same_v<EnvVar::value_type, uint64_t>);
    return EnvVar::Ref().GetValue();
}

/// updates the cached value of an environment variable
template <typename EnvVar, typename ValueType>
void UpdateEnvVar(EnvVar, const ValueType& val) 
{
    static_assert(std::is_same_v<EnvVar::value_type, ValueType>);
    EnvVar::Ref().UpdateValue(val);
}

} // namespace miopen

#endif
