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
#include <mutex>
#include <sstream>

namespace miopen {

struct RecordPositions
{
    std::streamoff begin = -1;
    std::streamoff end   = -1;
};

#define MIOPEN_VALIDATE_LOCK(lock)                       \
    do                                                   \
    {                                                    \
        if(!(lock))                                      \
            MIOPEN_THROW("Db lock has failed to lock."); \
    } while(false)

static std::chrono::seconds GetLockTimeout() { return std::chrono::seconds{60}; }

using exclusive_lock = std::unique_lock<LockFile>;
using shared_lock    = std::shared_lock<LockFile>;

RamDb::RamDb(std::string path, bool warn_if_unreadable_) : Db(path, warn_if_unreadable_) {}

RamDb& RamDb::GetCached(const std::string& path, bool warn_if_unreadable)
{
    static std::mutex mutex;
    static const std::lock_guard<std::mutex> lock{mutex};

    static auto instances = std::unordered_map<std::string, RamDb>{};
    const auto it         = instances.find(path);

    if(it != instances.end())
        return it->second;

    auto& instance = instances
                         .emplace(std::piecewise_construct,
                                  std::forward_as_tuple(path),
                                  std::forward_as_tuple(path, warn_if_unreadable))
                         .first->second;
    instance.Prefetch();
    return instance;
}

boost::optional<DbRecord> RamDb::FindRecord(const std::string& problem)
{
    Validate();
    return FindRecordUnsafe(problem);
}

bool RamDb::StoreRecord(const DbRecord& record)
{
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    if(!StoreRecordUnsafe(record))
        return false;

    Invalidate();
    return true;
}

bool RamDb::UpdateRecord(DbRecord& record)
{
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    if(!UpdateRecordUnsafe(record))
        return false;

    Invalidate();
    return true;
}

bool RamDb::RemoveRecord(const std::string& key)
{
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    if(!RemoveRecordUnsafe(key))
        return false;

    Invalidate();
    return true;
}

bool RamDb::Remove(const std::string& key, const std::string& id)
{
    const auto lock = exclusive_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    auto record = FindRecordUnsafe(key);
    if(!record)
        return false;
    bool erased = record->EraseValues(id);
    if(!erased)
        return false;

    Invalidate();
    return StoreRecordUnsafe(*record);
}

boost::optional<miopen::DbRecord> RamDb::FindRecordUnsafe(const std::string& problem)
{
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

    MIOPEN_LOG_I2("Looking for key " << problem << " in cache for file " << GetFileName());
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

void RamDb::Invalidate() { file_read_time = 0; }

void RamDb::Validate()
{
    const auto lock = shared_lock(GetLockFile(), GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    if(!boost::filesystem::exists(GetFileName()))
	{
        if(!cache.empty())
            Prefetch();
        return;
	}

    const auto file_mod_time = boost::filesystem::last_write_time(GetFileName());

    if(file_mod_time < file_read_time)
        return;

    Prefetch();
}

void RamDb::Prefetch()
{
    Measure("Prefetch", [this]() {
        file_read_time = std::time(nullptr);
        auto file      = std::ifstream{GetFileName()};

        if(!file)
        {
            const auto log_level =
                GetWarnIfUnreadable() ? LoggingLevel::Warning : LoggingLevel::Info;
            MIOPEN_LOG(log_level, "File is unreadable: " << GetFileName());
            return;
        }

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
    });
}
} // namespace miopen
