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
#include <iostream>
#include <sstream>
#include <type_traits>

#include <miopen/each_args.hpp>
#include <miopen/object.hpp>

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

#define MIOPEN_LOG_ENUM_EACH(x) std::pair<std::string, decltype(x)>(#x, x)
#ifdef _MSC_VER
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
    Default = 0, // Warning level for Release builds, Info for Debug builds.
    Quiet   = 1, // None logging messages (except those controlled by MIOPEN_ENABLE_LOGGING*).
    Fatal   = 2, // Fatal errors only (not used yet).
    Error   = 3, // Errors and fatals.
    Warning = 4, // All errors and warnings.
    Info    = 5, // All above plus information for debugging purposes.
    Info2   = 6, // All above  plus more detailed information for debugging.
    Trace   = 7  // The most detailed debugging messages
};

const char* LoggingLevelToCString(LoggingLevel level);
std::string LoggingPrefix();

/// \return true if level is enabled.
/// \param level - one of the values defined in LoggingLevel.
bool IsLogging(LoggingLevel level = LoggingLevel::Error);
bool IsLoggingCmd();
bool IsLoggingFunctionCalls();

template <class T>
auto LogObjImpl(T* x) -> decltype(get_object(*x))
{
    return get_object(*x);
}

inline void* LogObjImpl(void* x) { return x; }

inline const void* LogObjImpl(const void* x) { return x; }

#ifndef _MSC_VER
template <class T, typename std::enable_if<(std::is_pointer<T>{}), int>::type = 0>
std::ostream& LogParam(std::ostream& os, std::string name, const T& x)
{
    os << '\t' << name << " = ";
    if(x == nullptr)
        os << "nullptr";
    else
        os << LogObjImpl(x);
    return os;
}

template <class T, typename std::enable_if<(not std::is_pointer<T>{}), int>::type = 0>
std::ostream& LogParam(std::ostream& os, std::string name, const T& x)
{
    os << '\t' << name << " = " << get_object(x);
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

#define MIOPEN_LOG_FUNCTION(...)                                                        \
    do                                                                                  \
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
    while(false)
#else
#define MIOPEN_LOG_FUNCTION(...)
#endif

std::string LoggingParseFunction(const char* func, const char* pretty_func);

#define MIOPEN_LOG(level, ...)                                                               \
    do                                                                                       \
    {                                                                                        \
        if(miopen::IsLogging(level))                                                         \
        {                                                                                    \
            std::ostringstream miopen_log_ss;                                                \
            miopen_log_ss << miopen::LoggingPrefix() << LoggingLevelToCString(level) << " [" \
                          << miopen::LoggingParseFunction(__func__,            /* NOLINT */  \
                                                          __PRETTY_FUNCTION__) /* NOLINT */  \
                          << "] " << __VA_ARGS__ << std::endl;                               \
            std::cerr << miopen_log_ss.str();                                                \
        }                                                                                    \
    } while(false)

#define MIOPEN_LOG_E(...) MIOPEN_LOG(miopen::LoggingLevel::Error, __VA_ARGS__)
#define MIOPEN_LOG_W(...) MIOPEN_LOG(miopen::LoggingLevel::Warning, __VA_ARGS__)
#define MIOPEN_LOG_I(...) MIOPEN_LOG(miopen::LoggingLevel::Info, __VA_ARGS__)
#define MIOPEN_LOG_I2(...) MIOPEN_LOG(miopen::LoggingLevel::Info2, __VA_ARGS__)
#define MIOPEN_LOG_T(...) MIOPEN_LOG(miopen::LoggingLevel::Trace, __VA_ARGS__)

#define MIOPEN_LOG_DRIVER_CMD(...)                                                      \
    do                                                                                  \
    {                                                                                   \
        std::ostringstream miopen_driver_cmd_ss;                                        \
        miopen_driver_cmd_ss << miopen::LoggingPrefix() << "Command"                    \
                             << " [" << miopen::LoggingParseFunction(                   \
                                            __func__, __PRETTY_FUNCTION__) /* NOLINT */ \
                             << "] ./bin/MIOpenDriver " << __VA_ARGS__ << std::endl;    \
        std::cerr << miopen_driver_cmd_ss.str();                                        \
    } while(false)

} // namespace miopen

#endif
