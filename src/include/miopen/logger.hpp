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

#include <array>
#include <iostream>
#include <miopen/each_args.hpp>
#include <type_traits>

// Helper macros to output a cmdline argument for the driver
#define MIOPEN_DRIVER_CMD(op) \
    __func__ << ": "          \
             << "./bin/MIOpenDriver " << op

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
    for(auto&& p : values)
    {
        if(p.second == x)
        {
            os << p.first;
            return os;
        }
    }
    os << "Unknown: " << x;
    return os;
}

enum class LoggingLevel
{
    Default = 0, // WARNING for Release builds, INFO for Debug builds.
    Quiet,
    Fatal,
    Error,
    Warning,
    Info,
    Info2,
    Trace // E.g. messages output by MIOPEN_LOG_FUNCTION).
};

const char* LoggingLevelToCString(LoggingLevel level);

std::string PlatformName();

/// \return true if level is enabled.
/// \param level - one of the values defined in LoggingLevel.
int IsLogging(LoggingLevel level = LoggingLevel::Error);
bool IsLoggingCmd();

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
    os << name << " = ";
    if(x == nullptr)
        os << "nullptr";
    else
        os << LogObjImpl(x);
    return os;
}

template <class T, typename std::enable_if<(not std::is_pointer<T>{}), int>::type = 0>
std::ostream& LogParam(std::ostream& os, std::string name, const T& x)
{
    os << name << " = " << get_object(x);
    return os;
}
#define MIOPEN_LOG_FUNCTION_EACH(param) miopen::LogParam(std::cerr, #param, param) << std::endl;

#define MIOPEN_LOG_FUNCTION(...)                                                                \
    if(miopen::IsLogging(miopen::LoggingLevel::Trace))                                          \
    {                                                                                           \
        std::cerr << miopen::PlatformName() << ": " << __PRETTY_FUNCTION__ << "{" << std::endl; \
        MIOPEN_PP_EACH_ARGS(MIOPEN_LOG_FUNCTION_EACH, __VA_ARGS__)                              \
        std::cerr << "}" << std::endl;                                                          \
    }
#else
#define MIOPEN_LOG_FUNCTION(...)
#endif

/// \todo __PRETTY_FUNCTION__ is too verbose, __func_ it too short.
/// Shall we add filename (no path, no ext) prior __func__.
#define MIOPEN_LOG(level, ...)                                                                  \
    do                                                                                          \
    {                                                                                           \
        if(miopen::IsLogging(level))                                                            \
        {                                                                                       \
            std::cerr << miopen::PlatformName() << ": " << LoggingLevelToCString(level) << " [" \
                      << __func__ << "] " << __VA_ARGS__ << std::endl;                          \
        }                                                                                       \
    } while(false)

#define MIOPEN_LOG_E(...) MIOPEN_LOG(miopen::LoggingLevel::Error, __VA_ARGS__)
#define MIOPEN_LOG_W(...) MIOPEN_LOG(miopen::LoggingLevel::Warning, __VA_ARGS__)
#define MIOPEN_LOG_I(...) MIOPEN_LOG(miopen::LoggingLevel::Info, __VA_ARGS__)
#define MIOPEN_LOG_I2(...) MIOPEN_LOG(miopen::LoggingLevel::Info2, __VA_ARGS__)

} // namespace miopen

#endif
