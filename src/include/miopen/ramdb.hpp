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
#pragma once

#include <miopen/db.hpp>
#include <miopen/db_record.hpp>

#include <boost/optional.hpp>

#include <ctime>
#include <unordered_map>
#include <string>
#include <sstream>

namespace miopen {

class LockFile;
struct RecordPositions;

class RamDb : protected Db
{
    public:
    RamDb(std::string path, bool warn_if_unreadable_);

    RamDb(const RamDb&) = delete;
    RamDb(RamDb&&)      = delete;
    RamDb& operator=(const RamDb&) = delete;
    RamDb& operator=(RamDb&&) = delete;

    static RamDb& GetCached(const std::string& path, bool warn_if_unreadable);

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

    std::time_t file_read_time = 0;
    std::unordered_map<std::string, CacheItem> cache;

	boost::optional<miopen::DbRecord> FindRecordUnsafe(const std::string& problem);

	void Invalidate();
	void Validate();
    void Prefetch();
};

} // namespace miopen
