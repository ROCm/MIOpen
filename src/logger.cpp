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
#include <cstdlib>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>
#include <miopen/config.h>

namespace miopen {

/// Kept for backward compatibility for some time.
/// Enables all logging levels at once.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_ENABLE_LOGGING)
MIOPEN_DECLARE_ENV_VAR(MIOPEN_ENABLE_LOGGING_CMD)

MIOPEN_DECLARE_ENV_VAR(MIOPEN_LOG_LEVEL)

namespace {

inline bool operator!=(const int& lhs, const LoggingLevel& rhs)
{
    return lhs != static_cast<int>(rhs);
}
inline bool operator>=(const int& lhs, const LoggingLevel& rhs)
{
    return lhs >= static_cast<int>(rhs);
}

} // namespace

int IsLogging(const LoggingLevel level)
{
    if(miopen::IsEnabled(MIOPEN_ENABLE_LOGGING{}))
        return true;
    const int enabled_level = miopen::Value(MIOPEN_LOG_LEVEL{});
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
bool IsLoggingCmd() { return miopen::IsEnabled(MIOPEN_ENABLE_LOGGING_CMD{}); }

std::string PlatformName()
{
#if MIOPEN_BACKEND_OPENCL
    return "MIOpen(OpenCL)";
#elif MIOPEN_BACKEND_HIP
    return "MIOpen(HIP)";
#else
    return "MIOpen";
#endif
}

} // namespace miopen
