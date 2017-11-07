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

#include "miopen/db_manage.hpp"
#include "miopen/logger.hpp"
#include "miopen/env.hpp"

#include <iostream>

MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_PERF_DB_MODE)

#define MIOPEN_LOG_E(...) MIOPEN_LOG(miopen::LoggingLevel::Error, __VA_ARGS__)

namespace miopen {

namespace {

inline bool operator<=(const PerfDbMode& lhs, const int& rhs)
{
    return static_cast<int>(lhs) <= rhs;
}
inline bool operator<(const int& lhs, const PerfDbMode& rhs) { return lhs < static_cast<int>(rhs); }

const char* SearchModeToCString(const PerfDbMode mode)
{
    switch(mode)
    {
    case PerfDbMode::SearchIfNotFound: return "SearchIfNotFound";
    case PerfDbMode::EnforceSearchIfEntryNotFound: return "EnforceSearchIfEntryNotFound";
    case PerfDbMode::EnforceSearchAlways: return "EnforceSearchAlways";
    case PerfDbMode::CleanAllEntries: return "CleanAllEntries";
    default: return "<Unknown>";
    }
}

} // namespace

PerfDbMode GetSearchMode()
{
    static const int val        = miopen::Value(MIOPEN_DEBUG_PERF_DB_MODE{});
    static const bool ok        = (PerfDbMode::Begin <= val && val < PerfDbMode::End);
    static const PerfDbMode ret = ok ? static_cast<PerfDbMode>(val) : PerfDbMode::Default;
    if(!ok)
    {
        MIOPEN_LOG_E("Wrong MIOPEN_DEBUG_PERF_DB_MODE, resetting to default: " << ret);
    }
    return ret;
}

std::ostream& operator<<(std::ostream& os, const PerfDbMode sm)
{
    return os << SearchModeToCString(sm);
}

} // namespace miopen
