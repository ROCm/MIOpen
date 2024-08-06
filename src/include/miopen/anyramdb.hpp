/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#pragma once

#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/lock_file.hpp>

#include <boost/optional.hpp>
#include <boost/any.hpp>

#include <chrono>
#include <map>
#include <string>
#include <sstream>

namespace miopen {

class LockFile;

struct MIOPEN_INTERNALS_EXPORT AnyRamDb
{
    using TRecord = std::vector<boost::any>;

public:
    AnyRamDb(const fs::path& filename_)
        : filename(filename_), lock_file(LockFile::Get(LockFilePath(filename_))){};

    AnyRamDb(const AnyRamDb&) = delete;
    AnyRamDb(AnyRamDb&&)      = delete;
    AnyRamDb& operator=(const AnyRamDb&) = delete;
    AnyRamDb& operator=(AnyRamDb&&) = delete;

    static AnyRamDb& GetCached(const fs::path& path);

    boost::optional<AnyRamDb::TRecord> FindRecord(const std::string& problem);
    bool RemoveRecord(const std::string& key);
    bool StoreRecord(const std::string& problem, TRecord& record);

    template <class TProblem>
    boost::optional<TRecord> FindRecord(const TProblem& problem)
    {
        std::stringstream ss;
        problem.Serialize(ss);
        const auto key = ss.str();
        return FindRecord(key);
    }
    template <class T>
    inline bool StoreRecord(const T& problem_config, TRecord& record)
    {
        std::stringstream ss;
        problem_config.Serialize(ss);
        const auto key = ss.str();
        return StoreRecord(key, record);
    }

    template <class T>
    inline bool RemoveRecord(const T& problem_config)
    {
        std::stringstream ss;
        problem_config.Serialize(ss);
        const auto key = ss.str();
        return RemoveRecord(key);
    }

private:
    std::map<std::string, std::vector<boost::any>> cache;
    fs::path filename;
    LockFile& lock_file;
    boost::optional<TRecord> FindRecordUnsafe(const std::string& problem);
    void UpdateCacheEntryUnsafe(const std::string& key, const TRecord& value);
};

/// \todo This is modified copy of code from db.hpp. Make a proper fix.
template <>
// cppcheck-suppress noConstructor
class DbTimer<AnyRamDb>
{
    AnyRamDb& inner;

    template <class TFunc>
    static auto Measure(const std::string& funcName, TFunc&& func)
    {
        if(!miopen::IsLogging(LoggingLevel::Info2))
            return func();

        const auto start = std::chrono::high_resolution_clock::now();
        const auto ret   = func();
        const auto end   = std::chrono::high_resolution_clock::now();
        MIOPEN_LOG_I2("Db::" << funcName << " time: " << (end - start).count() * .000001f << " ms");
        return ret;
    }

public:
    template <class... TArgs>
    DbTimer(TArgs&&... args) : inner(AnyRamDb::GetCached(args...))
    {
    }

    template <class TProblem>
    auto FindRecord(const TProblem& problem)
    {
        return Measure("FindRecord", [&]() { return inner.FindRecord(problem); });
    }
    template <typename T, typename TRecord>
    bool StoreRecord(const T& problem_config, TRecord& record)
    {
        return Measure("StoreRecord", [&]() { return inner.StoreRecord(problem_config, record); });
    }

    template <class TProblem>
    bool RemoveRecord(const TProblem& problem)
    {
        return Measure("RemoveRecord", [&]() { return inner.RemoveRecord(problem); });
    }
};

} // namespace miopen
