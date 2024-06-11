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

#include <miopen/anyramdb.hpp>

#include <miopen/errors.hpp>
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

#define MIOPEN_VALIDATE_LOCK(lock)                       \
    do                                                   \
    {                                                    \
        if(!(lock))                                      \
            MIOPEN_THROW("Db lock has failed to lock."); \
    } while(false)

static std::chrono::seconds GetLockTimeout() { return std::chrono::seconds{60}; }

using exclusive_lock = std::unique_lock<LockFile>;

AnyRamDb& AnyRamDb::GetCached(const fs::path& path)
{
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock{mutex};

    static auto instances = std::map<fs::path, std::unique_ptr<AnyRamDb>>{};
    const auto it         = instances.find(path);

    if(it != instances.end())
        return *it->second;

    return *instances.emplace(path, std::make_unique<AnyRamDb>(path)).first->second;
}

boost::optional<AnyRamDb::TRecord> AnyRamDb::FindRecord(const std::string& problem)
{
    const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    return FindRecordUnsafe(problem);
}

bool AnyRamDb::StoreRecord(const std::string& problem, AnyRamDb::TRecord& record)
{
    const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    UpdateCacheEntryUnsafe(problem, record);
    return true;
}

bool AnyRamDb::RemoveRecord(const std::string& key)
{
    MIOPEN_LOG_I2("Trying to remove record at key " << key << " from cache for file " << filename);
    const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    cache.erase(key);
    return true;
}

boost::optional<AnyRamDb::TRecord> AnyRamDb::FindRecordUnsafe(const std::string& problem)
{
    MIOPEN_LOG_I2("Looking for key " << problem << " in cache for file " << filename);
    const auto it = cache.find(problem);

    if(it == cache.end())
        return boost::none;

    return it->second;
}

void AnyRamDb::UpdateCacheEntryUnsafe(const std::string& key, const TRecord& value)
{
    const auto it = cache.find(key);
    if(it != cache.end())
    {
        auto& item = it->second;
        item       = value;
    }
    else
    {
        cache.emplace(key, value);
    }
}

} // namespace miopen
