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

#include <miopen/ramdb.hpp>

#include <miopen/errors.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/logger.hpp>

#include <miopen/filesystem.hpp>

#include <chrono>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>

namespace miopen {

fs::path RamDb::GetTimeFilePath(const fs::path& path) { return path + ".time"; }

static ramdb_clock::time_point GetDbModificationTime(const fs::path& path)
{
    const auto time_file_path = RamDb::GetTimeFilePath(path);
    auto file                 = std::ifstream{time_file_path};

    if(!file.is_open())
        // Zero time from epoch, considering file ancient if time can't be retrieved.
        return {};

    ramdb_clock::rep time;
    file >> time;
    return ramdb_clock::time_point{ramdb_clock::duration{time}};
}

static void UpdateDbModificationTime(const fs::path& path)
{
    MIOPEN_LOG_I2("Updating db modification time for " << path);

    const auto time           = ramdb_clock::now().time_since_epoch();
    const auto time_file_path = RamDb::GetTimeFilePath(path);
    auto file                 = std::ofstream{time_file_path};

    if(!file)
    {
        MIOPEN_LOG_E("Cannot update database modification time: " << time_file_path);
        return;
    }

    file << time.count();
}

#define MIOPEN_VALIDATE_LOCK(lock)                       \
    do                                                   \
    {                                                    \
        if(!(lock))                                      \
            MIOPEN_THROW("Db lock has failed to lock."); \
    } while(false)

static std::chrono::seconds GetLockTimeout() { return std::chrono::seconds{60}; }

using exclusive_lock = std::unique_lock<LockFile>;

RamDb::RamDb(DbKinds db_kind_, const fs::path& path, bool is_system)
    : PlainTextDb(db_kind_, path, is_system)
{
}

RamDb& RamDb::GetCached(DbKinds db_kind_, const fs::path& path, bool is_system)
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock{mutex};

    // We don't have to store kind to properly index as different dbs would have different paths
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static auto instances = std::map<fs::path, std::unique_ptr<RamDb>>{};
    const auto it         = instances.find(path);

    if(it != instances.end())
        return *it->second;

    auto& instance =
        *instances.emplace(path, std::make_unique<RamDb>(db_kind_, path, is_system)).first->second;
    if constexpr(!DisableUserDbFileIO)
    {
        const auto prefetch_lock = exclusive_lock(instance.GetLockFile(), GetLockTimeout());
        MIOPEN_VALIDATE_LOCK(prefetch_lock);
        instance.Prefetch();
    }
    return instance;
}

boost::optional<DbRecord> RamDb::FindRecord(const std::string& problem)
{
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    if(!ValidateUnsafe())
    {
        MIOPEN_LOG_I2("RamDb file is newer than cache, prefetching");
        Prefetch();
    }

    return FindRecordUnsafe(problem);
}

bool RamDb::StoreRecord(const DbRecord& record)
{
    const auto& key = record.GetKey();
    MIOPEN_LOG_I2("Trying to store record at key " << key << " in cache for file "
                                                   << GetFileName());
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    if constexpr(!DisableUserDbFileIO)
    {
        if(!StoreRecordUnsafe(record))
            return false;
        UpdateDbModificationTime(GetFileName());
    }

#if MIOPEN_DB_CACHE_WRITE_THROUGH
    UpdateCacheEntryUnsafe(record);
#else
    Prefetch();
#endif
    return true;
}

bool RamDb::UpdateRecord(DbRecord& record)
{
    const auto& key = record.GetKey();
    MIOPEN_LOG_I2("Trying to update record at key " << key << " in cache for file "
                                                    << GetFileName());
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    if constexpr(!DisableUserDbFileIO)
    {
        if(!UpdateRecordUnsafe(record))
            return false;
        UpdateDbModificationTime(GetFileName());
    }

#if MIOPEN_DB_CACHE_WRITE_THROUGH
    UpdateCacheEntryUnsafe(record);
#else
    Prefetch();
#endif
    return true;
}

bool RamDb::RemoveRecord(const std::string& key)
{
    MIOPEN_LOG_I2("Trying to remove record at key " << key << " from cache for file "
                                                    << GetFileName());
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

#if MIOPEN_DB_CACHE_WRITE_THROUGH
    const auto is_valid = ValidateUnsafe();
#endif

    if constexpr(!DisableUserDbFileIO)
    {
        if(!RemoveRecordUnsafe(key))
            return false;
        UpdateDbModificationTime(GetFileName());
    }

#if MIOPEN_DB_CACHE_WRITE_THROUGH
    if(is_valid)
    {
        cache.erase(key);
        file_read_time = ramdb_clock::now();
    }
#else
    Prefetch();
#endif

    return true;
}

bool RamDb::Remove(const std::string& key, const std::string& id)
{
    MIOPEN_LOG_I2("Trying to remove value at key " << key << " and id " << id
                                                   << " from cache for file " << GetFileName());
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

#if MIOPEN_DB_CACHE_WRITE_THROUGH
    const auto is_valid = ValidateUnsafe();
#endif

    auto record = FindRecordUnsafe(key);

    if(!record || !record->EraseValues(id))
        return false;

    if constexpr(!DisableUserDbFileIO)
    {
        if(!StoreRecordUnsafe(*record))
            return false;
        UpdateDbModificationTime(GetFileName());
    }

#if MIOPEN_DB_CACHE_WRITE_THROUGH
    if(is_valid)
    {
        if(record->GetSize() == 0)
        {
            cache.erase(key);
        }
        else
        {
            auto it = cache.find(key);
            auto ss = std::ostringstream{};
            record->WriteIdsAndValues(ss);
            it->second.content = ss.str();
        }

        file_read_time = ramdb_clock::now();
    }
#else
    Prefetch();
#endif

    return true;
}

boost::optional<miopen::DbRecord> RamDb::FindRecordUnsafe(const std::string& problem)
{
    MIOPEN_LOG_I2("Looking for key " << problem << " in cache for file " << GetFileName());
    const auto it = cache.find(problem);

    if(it == cache.end())
        return boost::none;

    auto record = DbRecord{problem};

    if(!record.ParseContents(it->second.content))
    {
        MIOPEN_LOG_E("Error parsing payload under the key: "
                     << problem << " form file " << GetFileName() << "#" << it->second.line);
        MIOPEN_LOG_E("Contents: " << it->second.content);
        return boost::none;
    }

    return record;
}

template <class TFunc>
static void Measure(const std::string& funcName, TFunc&& func)
{
    if(!miopen::IsLogging(LoggingLevel::Info))
        func();

    const auto start = std::chrono::high_resolution_clock::now();
    func();
    const auto end = std::chrono::high_resolution_clock::now();
    MIOPEN_LOG_I("RamDb::" << funcName << " time: " << (end - start).count() * .000001f << " ms");
}

bool RamDb::ValidateUnsafe()
{
    if(DisableUserDbFileIO)
        return true;
    if(!fs::exists(GetFileName()))
        return cache.empty();
    const auto file_mod_time     = GetDbModificationTime(GetFileName());
    const auto validation_result = file_mod_time < file_read_time;
    MIOPEN_LOG_I2("DB file is " << (validation_result ? "older" : "newer")
                                << " than cache: " << file_mod_time.time_since_epoch().count()
                                << ", " << file_read_time.time_since_epoch().count());
    return validation_result;
}

void RamDb::Prefetch()
{
    if(DisableUserDbFileIO)
        MIOPEN_THROW("Prefetch should never happen with disabled File IO");

    Measure("Prefetch", [this]() {
        auto file = std::ifstream{GetFileName()};

        if(!file)
        {
            const auto log_level =
                IsWarningIfUnreadable() ? LoggingLevel::Warning : LoggingLevel::Info;
            MIOPEN_LOG(log_level, "File is unreadable: " << GetFileName());
            return;
        }

        cache.clear();
        auto line   = std::string{};
        auto n_line = 0;

        while(std::getline(file, line))
        {
            ++n_line;

            if(line.empty())
                continue;

            const auto key_size = line.find('=');
            const bool is_key   = (key_size != std::string::npos && key_size != 0);

            if(!is_key)
            {
                MIOPEN_LOG_E("Ill-formed record: key not found: " << GetFileName() << "#"
                                                                  << n_line);
                continue;
            }

            const auto key      = line.substr(0, key_size);
            const auto contents = line.substr(key_size + 1);

            cache.emplace(key, CacheItem{n_line, contents});
        }

        file_read_time = ramdb_clock::now();
    });
}

#if MIOPEN_DB_CACHE_WRITE_THROUGH
void RamDb::UpdateCacheEntryUnsafe(const DbRecord& record)
{
    const auto is_valid = ValidateUnsafe();

    if constexpr(!DisableUserDbFileIO)
        UpdateDbModificationTime(GetFileName());

    if(is_valid)
    {
        const auto& key = record.GetKey();
        const auto it   = cache.find(key);
        auto ss         = std::ostringstream{};
        record.WriteIdsAndValues(ss);

        if(it != cache.end())
        {
            auto& item   = it->second;
            item.content = ss.str();
        }
        else
        {
            cache.emplace(key, CacheItem{-1, ss.str()});
        }
        file_read_time = ramdb_clock::now();
    }
}
#endif

} // namespace miopen
