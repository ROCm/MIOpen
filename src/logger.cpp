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
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/config.h>

#include <cstdlib>
#include <chrono>
#include <ios>
#include <iomanip>

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h> /* For SYS_xxx definitions */
#endif

namespace miopen {

/// Enable logging of the most important function calls.
/// Name of envvar in a bit inadequate due to historical reasons.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_ENABLE_LOGGING)

/// Prints driver command lines into log.
/// Works from any application which uses the library.
/// Allows to reproduce library use cases using the driver instead of the actual application.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_ENABLE_LOGGING_CMD)

/// Prefix each log line with information which allows the user
/// to uniquiely identify log records printed from different processes
/// or threads. Useful for debugging multi-process/multi-threaded apps.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_ENABLE_LOGGING_MPMT)

/// Add timestamps to each log line.
/// Not useful  with multi-process/multi-threaded apps.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME)

/// See LoggingLevel in the header.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_LOG_LEVEL)

namespace debug {

bool LoggingQuiet = false; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)

} // namespace debug

/// Disable logging quieting.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_LOGGING_QUIETING_DISABLE)

namespace {

inline bool operator!=(const int& lhs, const LoggingLevel& rhs)
{
    return lhs != static_cast<int>(rhs);
}
inline bool operator==(const int& lhs, const LoggingLevel& rhs)
{
    return lhs == static_cast<int>(rhs);
}
inline bool operator>=(const int& lhs, const LoggingLevel& rhs)
{
    return lhs >= static_cast<int>(rhs);
}
inline bool operator>(const int& lhs, const LoggingLevel& rhs)
{
    return lhs > static_cast<int>(rhs);
}

/// Returns value which uniquiely identifies current process/thread
/// and can be printed into logs for MP/MT environments.
inline int GetProcessAndThreadId()
{
#ifdef __linux__
    // LWP is fine for identifying both processes and threads.
    return syscall(SYS_gettid); // NOLINT
#else
    return 0; // Not implemented.
#endif
}

inline float GetTimeDiff()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static auto prev = std::chrono::steady_clock::now();
    auto now         = std::chrono::steady_clock::now();
    auto rv =
        std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(now - prev).count();
    prev = now;
    return rv;
}

} // namespace

bool IsLoggingDebugQuiet()
{
    return debug::LoggingQuiet && !miopen::IsEnabled(MIOPEN_DEBUG_LOGGING_QUIETING_DISABLE{});
}

bool IsLoggingFunctionCalls()
{
    return miopen::IsEnabled(MIOPEN_ENABLE_LOGGING{}) && !IsLoggingDebugQuiet();
}

bool IsLogging(const LoggingLevel level, const bool disableQuieting)
{
    auto enabled_level = miopen::Value(MIOPEN_LOG_LEVEL{});
    if(IsLoggingDebugQuiet() && !disableQuieting)
    {
        // Disable all levels higher than fatal.
        if(enabled_level > LoggingLevel::DebugQuietMax || enabled_level == LoggingLevel::Default)
            enabled_level = static_cast<int>(LoggingLevel::DebugQuietMax);
    }
    if(enabled_level != LoggingLevel::Default)
        return enabled_level >= level;
#ifdef NDEBUG // Simplest way.
    return LoggingLevel::Warning >= level;
#else
    return LoggingLevel::Info >= level;
#endif
}

const char* LoggingLevelToCString(const LoggingLevel level)
{
    // Intentionally straightforward.
    // The most frequently used come first.
    if(level == LoggingLevel::Error)
        return "Error";
    else if(level == LoggingLevel::Warning)
        return "Warning";
    else if(level == LoggingLevel::Info)
        return "Info";
    else if(level == LoggingLevel::Info2)
        return "Info2";
    else if(level == LoggingLevel::Trace)
        return "Trace";
    else if(level == LoggingLevel::Default)
        return "Default";
    else if(level == LoggingLevel::Quiet)
        return "Quiet";
    else if(level == LoggingLevel::Fatal)
        return "Fatal";
    else
        return "<Unknown>";
}
bool IsLoggingCmd()
{
    return miopen::IsEnabled(MIOPEN_ENABLE_LOGGING_CMD{}) && !IsLoggingDebugQuiet();
}

std::string LoggingPrefix()
{
    std::stringstream ss;
    if(miopen::IsEnabled(MIOPEN_ENABLE_LOGGING_MPMT{}))
    {
        ss << GetProcessAndThreadId() << ' ';
    }
    ss << "MIOpen";
#if MIOPEN_BACKEND_OPENCL
    ss << "(OpenCL)";
#elif MIOPEN_BACKEND_HIP
    ss << "(HIP)";
#endif
    if(miopen::IsEnabled(MIOPEN_ENABLE_LOGGING_ELAPSED_TIME{}))
    {
        ss << std::fixed << std::setprecision(3) << std::setw(8) << GetTimeDiff();
    }
    ss << ": ";
    return ss.str();
}

/// Expected to be invoked with __func__ and __PRETTY_FUNCTION__.
std::string LoggingParseFunction(const char* func, const char* pretty_func)
{
    std::string fname{func};
    if(fname != "operator()")
        return fname;
    // lambda
    const std::string pf{pretty_func};
    const std::string pf_tail{pf.substr(0, pf.find_first_of('('))};
    return pf_tail.substr(1 + pf_tail.find_last_of(':'));
}

} // namespace miopen
