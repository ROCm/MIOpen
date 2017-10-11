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

namespace miopen {

/// Kept for backward compatibility for some time.
/// Enables all logging levels at once.
MIOPEN_DECLARE_ENV_VAR(MIOPEN_ENABLE_LOGGING)

/// 0 or undefined - Set logging level to the default:
///     3 for Release builds,
///     5 for Debug builds.
/// 1 - None log messages.
/// 2 - (Reserved for FATAL messages)
/// 3 - ERROR messages.
/// 4 - ...plus WARNINGs.
/// 5 - ...plus INFO messages.
/// 6 - ...plus TRACE information (e.g. output by MIOPEN_LOG_FUNCTION).
MIOPEN_DECLARE_ENV_VAR(MIOPEN_LOG_LEVEL)

int GetLoggingLevel()
{
    if(miopen::IsEnabled(MIOPEN_ENABLE_LOGGING{}))
        return 6;
    const int val = miopen::Value(MIOPEN_LOG_LEVEL{});
    if(val == 0)
    {
#ifdef NDEBUG // Simplest way.
        return 3;
#else
        return 5;
#endif
    }
    return val;
}

bool IsLoggingLevelError() { return GetLoggingLevel() >= 3; }
bool IsLoggingLevelWarning() { return GetLoggingLevel() >= 4; }
bool IsLoggingLevelInfo() { return GetLoggingLevel() >= 5; }
bool IsLoggingLevelTrace() { return GetLoggingLevel() >= 6; }

} // namespace miopen
