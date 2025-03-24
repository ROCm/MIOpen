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
#ifndef GUARD_MIOPEN_LOGGER_HPP
#define GUARD_MIOPEN_LOGGER_HPP

#include <algorithm>
#include <array>
#include <vector>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <chrono>

#include <miopen/each_args.hpp>
#include <miopen/object.hpp>
#include <miopen/config.hpp>

#if MIOPEN_USE_ROCTRACER
#include <roctracer/roctx.h>
#endif

// See https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms
#define MIOPEN_PP_CAT(x, y) MIOPEN_PP_PRIMITIVE_CAT(x, y)
#define MIOPEN_PP_PRIMITIVE_CAT(x, y) x##y

#define MIOPEN_PP_IIF(c) MIOPEN_PP_PRIMITIVE_CAT(MIOPEN_PP_IIF_, c)
#define MIOPEN_PP_IIF_0(t, ...) __VA_ARGS__
#define MIOPEN_PP_IIF_1(t, ...) t

#define MIOPEN_PP_IS_PAREN(x) MIOPEN_PP_IS_PAREN_CHECK(MIOPEN_PP_IS_PAREN_PROBE x)
#define MIOPEN_PP_IS_PAREN_CHECK(...) MIOPEN_PP_IS_PAREN_CHECK_N(__VA_ARGS__, 0)
#define MIOPEN_PP_IS_PAREN_PROBE(...) ~, 1,
#define MIOPEN_PP_IS_PAREN_CHECK_N(x, n, ...) n

#define MIOPEN_PP_EAT(...)
#define MIOPEN_PP_EXPAND(...) __VA_ARGS__
#define MIOPEN_PP_COMMA(...) ,

#define MIOPEN_PP_TRANSFORM_ARGS(m, ...)                                 \
    MIOPEN_PP_EXPAND(MIOPEN_PP_PRIMITIVE_TRANSFORM_ARGS(m,               \
                                                        MIOPEN_PP_COMMA, \
                                                        __VA_ARGS__,     \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        (),              \
                                                        ()))

#define MIOPEN_PP_EACH_ARGS(m, ...)                                    \
    MIOPEN_PP_EXPAND(MIOPEN_PP_PRIMITIVE_TRANSFORM_ARGS(m,             \
                                                        MIOPEN_PP_EAT, \
                                                        __VA_ARGS__,   \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        (),            \
                                                        ()))

#define MIOPEN_PP_PRIMITIVE_TRANSFORM_ARGS(                                              \
    m, delim, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, ...) \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x0)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x1)                                         \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x1)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x2)                                         \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x2)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x3)                                         \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x3)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x4)                                         \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x4)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x5)                                         \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x5)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x6)                                         \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x6)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x7)                                         \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x7)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x8)                                         \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x8)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x9)                                         \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x9)                                             \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x10)                                        \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x10)                                            \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x11)                                        \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x11)                                            \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x12)                                        \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x12)                                            \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x13)                                        \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x13)                                            \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x14)                                        \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x14)                                            \
    MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(delim, x15) MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x15)

#define MIOPEN_PP_PRIMITIVE_TRANSFORM_ARG(m, x) \
    MIOPEN_PP_IIF(MIOPEN_PP_IS_PAREN(x))(MIOPEN_PP_EAT, m)(x)

namespace miopen {

template <class Range>
std::ostream& LogRange(std::ostream& os, Range&& r, std::string delim)
{
    bool first = true;
    for(auto&& x : r)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << x;
    }
    return os;
}

template <class T, class... Ts>
std::array<T, sizeof...(Ts) + 1> make_array(T x, Ts... xs)
{
    return {{x, xs...}};
}

// MSVC's preprocessor and CPPCHECK seem unable
// to properly handle some complex stuff. We have to disable
// some debugging features to avoid build errors.
#define WORKAROUND_ISSUE_PP_TRANSFORM_ARGS 0
#if defined(_MSC_VER) || defined(CPPCHECK)
#undef WORKAROUND_ISSUE_PP_TRANSFORM_ARGS
#define WORKAROUND_ISSUE_PP_TRANSFORM_ARGS 1
#endif

#define MIOPEN_LOG_ENUM_EACH(x) std::pair<std::string, decltype(x)>(#x, x)
#if WORKAROUND_ISSUE_PP_TRANSFORM_ARGS
#define MIOPEN_LOG_ENUM(os, ...) os
#else
#define MIOPEN_LOG_ENUM(os, x, ...) \
    miopen::LogEnum(os, x, make_array(MIOPEN_PP_TRANSFORM_ARGS(MIOPEN_LOG_ENUM_EACH, __VA_ARGS__)))
#endif

template <class T, class Range>
std::ostream& LogEnum(std::ostream& os, T x, Range&& values)
{
    auto it = std::find_if(values.begin(), values.end(), [&](auto&& p) { return p.second == x; });
    if(it == values.end())
        os << "Unknown: " << x;
    else
        os << it->first;
    return os;
}

enum class LoggingLevel
{
    Default       = 0, // Warning level for Release builds, Info for Debug builds.
    Quiet         = 1, // None logging messages (except those controlled by MIOPEN_ENABLE_LOGGING*).
    Fatal         = 2, // Fatal errors only (not used yet).
    Error         = 3, // Errors and fatals.
    Warning       = 4, // All errors and warnings.
    Info          = 5, // All above plus information for debugging purposes.
    Info2         = 6, // All above  plus more detailed information for debugging.
    Trace         = 7, // The most detailed debugging messages
    DebugQuietMax = Error
};

// Warnings in installable builds, errors otherwise.
constexpr const LoggingLevel LogWELevel =
    MIOPEN_INSTALLABLE ? miopen::LoggingLevel::Warning : miopen::LoggingLevel::Error;

namespace debug {

/// Quiet mode for debugging/testing purposes. All logging (including MIOPEN_ENABLE_LOGGING*)
/// is switched OFF unless it happens at DebugQuietMax or higher priority level, OR invoked
/// by MIOPEN_LOG_NQ* macros (that ignore this switch).
///
/// WARNING: This switch is not intended for use in multi-threaded applications.
MIOPEN_EXPORT extern bool
    LoggingQuiet; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

} // namespace debug

MIOPEN_INTERNALS_EXPORT std::string LoggingLevelToCustomString(LoggingLevel level,
                                                               const char* custom);
MIOPEN_INTERNALS_EXPORT const char* LoggingLevelToCString(LoggingLevel level);
MIOPEN_INTERNALS_EXPORT std::string LoggingPrefix();

/// \return true if level is enabled.
/// \param level - one of the values defined in LoggingLevel.
MIOPEN_INTERNALS_EXPORT bool IsLogging(LoggingLevel level, bool disableQuieting = false);
bool IsLoggingCmd();
bool IsLoggingFunctionCalls();
#if MIOPEN_USE_ROCTRACER
bool IsLoggingToRoctx();
#endif

namespace logger {

template <typename T, typename S>
struct CArray
{
    std::vector<T> values;
    CArray(const T* x, S size)
    {
        if(x != nullptr && size > 0)
            values = {x, x + static_cast<std::size_t>(size)};
    }
};

} // namespace logger

template <class T>
auto LogObjImpl(T* x) -> decltype(get_object(*x))
{
    return get_object(*x);
}

inline void* LogObjImpl(void* x) { return x; }

inline const void* LogObjImpl(const void* x) { return x; }

#if !WORKAROUND_ISSUE_PP_TRANSFORM_ARGS
template <class T, typename std::enable_if<(std::is_pointer<T>{}), int>::type = 0>
std::ostream& LogParam(std::ostream& os, std::string name, const T& x, bool indent = true)
{
    if(indent)
        os << '\t';
    os << name << " = ";
    if(x == nullptr)
        os << "nullptr";
    else
        os << LogObjImpl(x);
    return os;
}

template <class T, typename std::enable_if<(not std::is_pointer<T>{}), int>::type = 0>
std::ostream& LogParam(std::ostream& os, std::string name, const T& x, bool indent = true)
{
    if(indent)
        os << '\t';
    os << name << " = " << get_object(x);
    return os;
}

template <class T>
std::ostream&
LogParam(std::ostream& os, std::string name, const std::vector<T>& vec, bool indent = true)
{
    if(indent)
        os << '\t';
    os << name << " = { ";
    for(auto& val : vec)
        os << val << ' ';
    os << '}';
    return os;
}

#define MIOPEN_LOG_FUNCTION_EACH(param)                                         \
    do                                                                          \
    {                                                                           \
        /* Clear temp stringstream & reset its state: */                        \
        std::ostringstream().swap(miopen_log_func_ss);                          \
        /* Use stringstram as ostream to engage existing template functions: */ \
        std::ostream& miopen_log_func_ostream = miopen_log_func_ss;             \
        miopen_log_func_ostream << miopen::LoggingPrefix();                     \
        miopen::LogParam(miopen_log_func_ostream, #param, param) << std::endl;  \
        std::cerr << miopen_log_func_ss.str();                                  \
    } while(false);

#define MIOPEN_LOG_FUNCTION_EACH_ROCTX(param)                                     \
    do                                                                            \
    {                                                                             \
        /* Use stringstram as ostream to engage existing template functions: */   \
        std::ostream& miopen_log_func_ostream = miopen_log_func_ss;               \
        miopen::LogParam(miopen_log_func_ostream, #param, param, false) << " | "; \
    } while(false);

#if MIOPEN_USE_ROCTRACER
#define MIOPEN_LOG_ROCTX_DEFINE_OBJECT miopen::LogScopeRoctx logtx;
#define MIOPEN_LOG_ROCTX_DO_LOGGING(...)                                 \
    if(miopen::IsLoggingToRoctx())                                       \
    {                                                                    \
        std::ostringstream miopen_log_func_ss;                           \
        miopen_log_func_ss << "s_api = " << __FUNCTION__ << " | ";       \
        MIOPEN_PP_EACH_ARGS(MIOPEN_LOG_FUNCTION_EACH_ROCTX, __VA_ARGS__) \
        logtx.logRange(miopen_log_func_ss.str());                        \
    }
#else
#define MIOPEN_LOG_ROCTX_DEFINE_OBJECT
#define MIOPEN_LOG_ROCTX_DO_LOGGING(...)
#endif

#define MIOPEN_LOG_FUNCTION(...)                                                        \
    MIOPEN_LOG_ROCTX_DEFINE_OBJECT                                                      \
    do                                                                                  \
    {                                                                                   \
        if(miopen::IsLoggingFunctionCalls())                                            \
        {                                                                               \
            std::ostringstream miopen_log_func_ss;                                      \
            miopen_log_func_ss << miopen::LoggingPrefix() << __PRETTY_FUNCTION__ << "{" \
                               << std::endl;                                            \
            std::cerr << miopen_log_func_ss.str();                                      \
            MIOPEN_PP_EACH_ARGS(MIOPEN_LOG_FUNCTION_EACH, __VA_ARGS__)                  \
            std::ostringstream().swap(miopen_log_func_ss);                              \
            miopen_log_func_ss << miopen::LoggingPrefix() << "}" << std::endl;          \
            std::cerr << miopen_log_func_ss.str();                                      \
        }                                                                               \
        MIOPEN_LOG_ROCTX_DO_LOGGING(__VA_ARGS__)                                        \
    } while(false)
#else
#define MIOPEN_LOG_FUNCTION(...)
#endif

constexpr std::string_view LoggingParseFunction(const std::string_view func,
                                                const std::string_view pretty_func)
{
    if(func != "operator()")
        return func;
    // lambda
    const auto pf_tail = pretty_func.substr(0, pretty_func.find_first_of('('));
    return pf_tail.substr(1 + pf_tail.find_last_of(':'));
}

#ifdef _MSC_VER
#define MIOPEN_GET_FN_NAME miopen::LoggingParseFunction(__func__, __FUNCSIG__)
#else
#define MIOPEN_GET_FN_NAME miopen::LoggingParseFunction(__func__, __PRETTY_FUNCTION__)
#endif

#define MIOPEN_LOG_XQ_CUSTOM(level, disableQuieting, category, fn_name, ...)                \
    do                                                                                      \
    {                                                                                       \
        if(miopen::IsLogging(level, disableQuieting))                                       \
        {                                                                                   \
            std::ostringstream miopen_log_ss;                                               \
            miopen_log_ss << miopen::LoggingPrefix() << category << " [" << fn_name << "] " \
                          << __VA_ARGS__ << std::endl;                                      \
            std::cerr << miopen_log_ss.str();                                               \
        }                                                                                   \
    } while(false)

#define MIOPEN_LOG_XQ_(level, disableQuieting, fn_name, ...) \
    MIOPEN_LOG_XQ_CUSTOM(level, disableQuieting, LoggingLevelToCString(level), fn_name, __VA_ARGS__)

#define MIOPEN_LOG_CUSTOM(level, category, ...)                       \
    MIOPEN_LOG_XQ_CUSTOM(level,                                       \
                         false,                                       \
                         LoggingLevelToCustomString(level, category), \
                         MIOPEN_GET_FN_NAME,                          \
                         __VA_ARGS__)
#define MIOPEN_LOG(level, ...) MIOPEN_LOG_XQ_(level, false, MIOPEN_GET_FN_NAME, __VA_ARGS__)
#define MIOPEN_LOG_NQ_(level, ...) MIOPEN_LOG_XQ_(level, true, MIOPEN_GET_FN_NAME, __VA_ARGS__)

#define MIOPEN_LOG_E(...) MIOPEN_LOG(miopen::LoggingLevel::Error, __VA_ARGS__)
#define MIOPEN_LOG_E_FROM(from, ...) \
    MIOPEN_LOG_XQ_(miopen::LoggingLevel::Error, false, from, __VA_ARGS__)
#define MIOPEN_LOG_W(...) MIOPEN_LOG(miopen::LoggingLevel::Warning, __VA_ARGS__)
#define MIOPEN_LOG_I(...) MIOPEN_LOG(miopen::LoggingLevel::Info, __VA_ARGS__)
#define MIOPEN_LOG_I2(...) MIOPEN_LOG(miopen::LoggingLevel::Info2, __VA_ARGS__)
#define MIOPEN_LOG_T(...) MIOPEN_LOG(miopen::LoggingLevel::Trace, __VA_ARGS__)

// These are always shown (do not obey the miopen::debug::LoggingQuiet switch).
#define MIOPEN_LOG_NQE(...) MIOPEN_LOG_NQ_(miopen::LoggingLevel::Error, __VA_ARGS__)
#define MIOPEN_LOG_NQI(...) MIOPEN_LOG_NQ_(miopen::LoggingLevel::Info, __VA_ARGS__)
#define MIOPEN_LOG_NQI2(...) MIOPEN_LOG_NQ_(miopen::LoggingLevel::Info2, __VA_ARGS__)

// Warnings in installable builds, errors otherwise.
#define MIOPEN_LOG_WE(...) MIOPEN_LOG(LogWELevel, __VA_ARGS__)

#define MIOPEN_LOG_DRIVER_COMMAND(driver, ...)                                               \
    do                                                                                       \
    {                                                                                        \
        std::ostringstream miopen_driver_cmd_ss;                                             \
        miopen_driver_cmd_ss << miopen::LoggingPrefix() << "Command"                         \
                             << " [" << MIOPEN_GET_FN_NAME << "] " driver " " << __VA_ARGS__ \
                             << std::endl;                                                   \
        std::cerr << miopen_driver_cmd_ss.str();                                             \
    } while(false)

#ifdef _WIN32
#define MIOPEN_LOG_DRIVER_CMD(...) MIOPEN_LOG_DRIVER_COMMAND("MIOpenDriver.exe", __VA_ARGS__)
#else
#define MIOPEN_LOG_DRIVER_CMD(...) MIOPEN_LOG_DRIVER_COMMAND("./bin/MIOpenDriver", __VA_ARGS__)
#endif

#if MIOPEN_LOG_FUNC_TIME_ENABLE
class LogScopeTime
{
public:
    LogScopeTime(std::string name)
        : m_name(std::move(name)), m_beg(std::chrono::high_resolution_clock::now())
    {
    }
    ~LogScopeTime()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - m_beg);
        MIOPEN_LOG_I2(m_name << " : " << dur.count() << " us");
    }

private:
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
};

#define MIOPEN_LOG_SCOPE_TIME const miopen::LogScopeTime miopen_timer(MIOPEN_GET_FN_NAME)
#else
#define MIOPEN_LOG_SCOPE_TIME
#endif

#if MIOPEN_USE_ROCTRACER
class LogScopeRoctx
{
public:
    LogScopeRoctx() = default;
    explicit LogScopeRoctx(const std::string& name) { logRange(name); }
    void logRange(const std::string& name)
    {
        if(!m_active)
        {
            roctxRangePush(name.c_str());
            m_active = true;
        }
    }
    ~LogScopeRoctx()
    {
        if(m_active)
        {
            roctxRangePop();
        }
    }

private:
    bool m_active{false};
};
#endif

} // namespace miopen

#endif
