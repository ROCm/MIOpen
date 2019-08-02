/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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
#include <miopen/md5.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem.hpp>

#include <chrono>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>

namespace miopen {

static ramdb_clock::time_point GetDbModificationTime(const std::string& path)
{
    const auto time_file_path = path + ".time";
    auto file                 = std::ifstream{time_file_path};

    if(!file)
        // Zero time from epoch, considering file ancient if time can't be retrieved.
        return {};

    ramdb_clock::rep time;
    file >> time;
    return ramdb_clock::time_point{ramdb_clock::duration{time}};
}

static void UpdateDbModificationTime(const std::string& path)
{
    MIOPEN_LOG_I2("Updating db modification time for " << path);

    const auto time           = ramdb_clock::now().time_since_epoch();
    const auto time_file_path = path + ".time";
    auto file                 = std::ofstream{time_file_path};

    if(!file)
    {
        MIOPEN_LOG_E("Cannot update database modification time: " + time_file_path);
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

RamDb::RamDb(std::string path, bool warn_if_unreadable_)
    : Db(path, warn_if_unreadable_), file_read_time()
{
}

RamDb& RamDb::GetCached(const std::string& path, bool warn_if_unreadable)
{
    static std::mutex mutex;
    static const std::lock_guard<std::mutex> lock{mutex};

    static auto instances = std::map<std::string, RamDb>{};

    {
        const auto it = instances.find(path);
        if(it != instances.end())
            return it->second;
    }

    // map::emplace(std::piecewise_construct, ...) is constructing map entry in-place allowing use
    // of non-copiable non-movable type as key/value type which RamDb is.
    auto emplace_ret = instances.emplace(std::piecewise_construct,
                                         // This are parameters of key ctor:
                                         std::forward_as_tuple(path),
                                         // This are parameters of value ctor:
                                         std::forward_as_tuple(path, warn_if_unreadable));
    const auto it  = emplace_ret.first;
    auto& instance = it->second;

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

    if(!StoreRecordUnsafe(record))
        return false;

    UpdateDbModificationTime(GetFileName());
    // UpdateCacheEntryUnsafe(record);
    return true;
}

bool RamDb::UpdateRecord(DbRecord& record)
{
    const auto& key = record.GetKey();
    MIOPEN_LOG_I2("Trying to update record at key " << key << " in cache for file "
                                                    << GetFileName());
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    if(!UpdateRecordUnsafe(record))
        return false;

    UpdateDbModificationTime(GetFileName());
    // UpdateCacheEntryUnsafe(record);
    return true;
}

bool RamDb::RemoveRecord(const std::string& key)
{
    MIOPEN_LOG_I2("Trying to remove record at key " << key << " from cache for file "
                                                    << GetFileName());
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    // const auto is_valid = ValidateUnsafe();
    if(!RemoveRecordUnsafe(key))
        return false;

    UpdateDbModificationTime(GetFileName());

    /*if(is_valid)
    {
        cache.erase(key);
        file_read_time = ramdb_clock::now();
    }*/

    return true;
}

bool RamDb::Remove(const std::string& key, const std::string& id)
{
    MIOPEN_LOG_I2("Trying to remove value at key " << key << " and id " << id
                                                   << " from cache for file "
                                                   << GetFileName());
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    // const auto is_valid = ValidateUnsafe();
    auto record = FindRecordUnsafe(key);
    if(!record || !record->EraseValues(id) || !StoreRecordUnsafe(*record))
        return false;

    UpdateDbModificationTime(GetFileName());

    /*if(is_valid)
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
    }*/

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
        MIOPEN_LOG_E("Error parsing payload under the key: " << problem << " form file "
                                                             << GetFileName()
                                                             << "#"
                                                             << it->second.line);
        MIOPEN_LOG_E("Contents: " << it->second.content);
        return boost::none;
    }

    return record;
}

template <class TFunc>
static auto Measure(const std::string& funcName, TFunc&& func)
{
    if(!miopen::IsLogging(LoggingLevel::Info))
        return func();

    const auto start = std::chrono::high_resolution_clock::now();
    func();
    const auto end = std::chrono::high_resolution_clock::now();
    MIOPEN_LOG_I("RamDb::" << funcName << " time: " << (end - start).count() * .000001f << " ms");
}

bool RamDb::ValidateUnsafe()
{
    if(!boost::filesystem::exists(GetFileName()))
        return cache.empty();
    const auto file_mod_time     = GetDbModificationTime(GetFileName());
    const auto validation_result = file_mod_time < file_read_time;
    MIOPEN_LOG_I2("DB file is " << (validation_result ? "older" : "newer") << " than cache: "
                                << file_mod_time.time_since_epoch().count()
                                << ", "
                                << file_read_time.time_since_epoch().count());
    return validation_result;
}

void RamDb::Prefetch()
{
    Measure("Prefetch", [this]() {
        auto file = std::ifstream{GetFileName()};

        if(!file)
        {
            const auto log_level =
                GetWarnIfUnreadable() ? LoggingLevel::Warning : LoggingLevel::Info;
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

/*void RamDb::UpdateCacheEntryUnsafe(const DbRecord& record)
{
    const auto is_valid = ValidateUnsafe();
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
}*/

} // namespace miopen
