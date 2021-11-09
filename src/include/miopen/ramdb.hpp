/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <boost/optional.hpp>

#include <chrono>
#include <map>
#include <string>
#include <sstream>

// Value of one enables experimental write-through feature of RamDb.
// It provides some performance gain in case of multi-threaded cache write operations.
#define MIOPEN_DB_CACHE_WRITE_THROUGH 1

namespace miopen {

using ramdb_clock = std::chrono::steady_clock;

class LockFile;

class RamDb : protected PlainTextDb
{
    public:
    RamDb(std::string path, bool is_system, const std::string& /*arch*/, std::size_t /*num_cu*/)
        : RamDb(path, is_system)
    {
    }

    RamDb(std::string path, bool is_system = false);

    RamDb(const RamDb&) = delete;
    RamDb(RamDb&&)      = delete;
    RamDb& operator=(const RamDb&) = delete;
    RamDb& operator=(RamDb&&) = delete;

    static std::string GetTimeFilePath(const std::string& path);
    static RamDb& GetCached(const std::string& path, bool is_system);

    static RamDb& GetCached(const std::string& path,
                            bool is_system,
                            const std::string& /*arch*/,
                            std::size_t /*num_cu*/)
    {
        return GetCached(path, is_system);
    }

    boost::optional<DbRecord> FindRecord(const std::string& problem);

    template <class TProblem>
    boost::optional<DbRecord> FindRecord(const TProblem& problem)
    {
        const auto key = DbRecord::Serialize(problem);
        return FindRecord(key);
    }

    template <class TProblem, class TValue>
    bool Load(const TProblem& problem, const std::string& id, TValue& value)
    {
        const auto record = FindRecord(problem);
        if(!record)
            return false;
        return record->GetValues(id, value);
    }

    bool StoreRecord(const DbRecord& record);
    bool UpdateRecord(DbRecord& record);
    bool RemoveRecord(const std::string& key);
    bool Remove(const std::string& key, const std::string& id);

    template <class T>
    inline bool Remove(const T& problem_config, const std::string& id)
    {
        const auto key = DbRecord::Serialize(problem_config);
        return Remove(key, id);
    }

    template <class T>
    inline bool RemoveRecord(const T& problem_config)
    {
        const auto key = DbRecord::Serialize(problem_config);
        return RemoveRecord(key);
    }

    template <class T, class V>
    inline boost::optional<DbRecord>
    Update(const T& problem_config, const std::string& id, const V& values)
    {
        DbRecord record(problem_config);
        record.SetValues(id, values);
        const auto ok = UpdateRecord(record);
        if(ok)
            return record;
        else
            return boost::none;
    }

    private:
    struct CacheItem
    {
        int line;
        std::string content;
    };

    ramdb_clock::time_point file_read_time;
    std::map<std::string, CacheItem> cache;

    boost::optional<miopen::DbRecord> FindRecordUnsafe(const std::string& problem);

    bool ValidateUnsafe();
    void Prefetch();

#if MIOPEN_DB_CACHE_WRITE_THROUGH
    void UpdateCacheEntryUnsafe(const DbRecord& record);
#endif
};

/// \todo This is modified copy of code from db.hpp. Make a proper fix.
template <>
// cppcheck-suppress noConstructor
class DbTimer<RamDb>
{
    RamDb& inner;

    template <class TFunc>
    static auto Measure(const std::string& funcName, TFunc&& func)
    {
        if(!miopen::IsLogging(LoggingLevel::Info2))
            return func();

        const auto start = std::chrono::high_resolution_clock::now();
        auto ret         = func();
        const auto end   = std::chrono::high_resolution_clock::now();
        MIOPEN_LOG_I2("Db::" << funcName << " time: " << (end - start).count() * .000001f << " ms");
        return ret;
    }

    public:
    template <class... TArgs>
    DbTimer(TArgs&&... args) : inner(RamDb::GetCached(args...))
    {
    }

    template <class TProblem>
    auto FindRecord(const TProblem& problem)
    {
        return Measure("FindRecord", [&]() { return inner.FindRecord(problem); });
    }

    bool StoreRecord(const DbRecord& record)
    {
        return Measure("StoreRecord", [&]() { return inner.StoreRecord(record); });
    }

    bool UpdateRecord(DbRecord& record)
    {
        return Measure("UpdateRecord", [&]() { return inner.UpdateRecord(record); });
    }

    template <class TProblem>
    bool RemoveRecord(const TProblem& problem)
    {
        return Measure("RemoveRecord", [&]() { return inner.RemoveRecord(problem); });
    }

    template <class TProblem, class TValue>
    auto Update(const TProblem& problem, const std::string& id, const TValue& value)
    {
        return Measure("Update", [&]() { return inner.Update(problem, id, value); });
    }

    template <class TProblem, class TValue>
    bool Load(const TProblem& problem, const std::string& id, TValue& value)
    {
        return Measure("Load", [&]() { return inner.Load(problem, id, value); });
    }

    template <class TProblem>
    bool Remove(const TProblem& problem, const std::string& id)
    {
        return Measure("Remove", [&]() { return inner.Remove(problem, id); });
    }
};

} // namespace miopen
