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
#ifndef GUARD_MIOPEN_DB_HPP_
#define GUARD_MIOPEN_DB_HPP_

#include <string>

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/optional.hpp>

#include <miopen/db_record.hpp>
#include <miopen/lock_file.hpp>

namespace miopen {
/// No instance of this class should be used from several threads at the same time.
class Db
{
    public:
    inline Db(const std::string& filename_)
        : filename(filename_), lock_file(LockFileDispatcher::Get((filename_ + ".lock").c_str()))
    {
    }

    /// Searches db for provided key and returns found record or none if key not found in database
    inline boost::optional<DbRecord> FindRecord(const std::string& key)
    {
        shared_lock lock(lock_file);
        return FindRecordUnsafe(key, nullptr);
    }

    template <class T>
    inline boost::optional<DbRecord> FindRecord(const T& problem_config)
    {
        shared_lock lock(lock_file);
        return FindRecordUnsafe(problem_config);
    }

    /// Stores provided record in database. If record with same key is already in database it is
    /// replaced by provided record.
    ///
    /// Returns true if store was successful, false otherwise.
    inline bool StoreRecord(const DbRecord& record)
    {
        exclusive_lock lock(lock_file);
        return StoreRecordUnsafe(record);
    }

    /// Stores provided record in database. If record with same key is already in database it is
    /// updated with values from provided record. Provided records data is also updated via
    /// DbRecord::Merge().
    ///
    /// Returns true if update was successful, false otherwise.
    inline bool UpdateRecord(DbRecord& record)
    {
        exclusive_lock lock(lock_file);
        return UpdateRecordUnsafe(record);
    }

    /// Removes record with provided key from db
    ///
    /// Returns true if remove was successful, false otherwise.
    inline bool RemoveRecord(const std::string& key)
    {
        exclusive_lock lock(lock_file);
        return RemoveRecordUnsafe(key);
    }

    template <class T>
    inline bool RemoveRecord(const T& problem_config)
    {
        auto key = DbRecord::Serialize(problem_config);
        return RemoveRecord(key);
    }

    /// Updates record under key PROBLEM_CONFIG with data ID:VALUES in database.
    /// Both T and V classes should have "void Serialize(std::ostream&) const" member function
    /// available.
    ///
    /// Returns updated record or none if update was unsuccessful.
    template <class T, class V>
    inline boost::optional<DbRecord>
    Update(const T& problem_config, const std::string& id, const V& values)
    {
        DbRecord record(problem_config);
        record.SetValues(id, values);
        bool ok = UpdateRecord(record);
        if(ok)
            return record;
        else
            return boost::none;
    }

    /// Searches for record with key PROBLEM_CONFIG and gets VALUES under the ID from it.
    /// Class T should have "void Serialize(std::ostream&) const" member function available.
    /// Class V shall have "bool Deserialize(const std::string& str)" member function available.
    ///
    /// Returns false if there is none PROBLEM_CONFIG=ID:VALUES in the database
    /// or in case of any error, e.g. if VALUES cannot be deserialized due to incorrect format.
    template <class T, class V>
    inline bool Load(const T& problem_config, const std::string& id, V& values)
    {
        auto record = FindRecord(problem_config);

        if(!record)
            return false;
        return record->GetValues(id, values);
    }

    /// Removes ID with associated VALUES from record with key PROBLEM_CONFIG from db.
    /// If payload of a record becomes empty after that, also removes the entire record
    ///
    /// Returns true if remove was successful. Returns false if this PROBLEM_CONFIG or ID was not
    /// found.
    template <class T>
    inline bool Remove(const T& problem_config, const std::string& id)
    {
        exclusive_lock lock(lock_file);
        auto record = FindRecordUnsafe(problem_config);
        if(!record)
            return false;
        bool erased = record->EraseValues(id);
        if(!erased)
            return false;
        return StoreRecordUnsafe(*record);
    }

    private:
    struct RecordPositions
    {
        std::streamoff begin = -1;
        std::streamoff end   = -1;
    };

    using exclusive_lock = boost::interprocess::scoped_lock<LockFile>;
    using shared_lock    = boost::interprocess::sharable_lock<LockFile>;

    std::string filename;
    LockFile lock_file;

    boost::optional<DbRecord> FindRecordUnsafe(const std::string& key, RecordPositions* pos);
    bool FlushUnsafe(const DbRecord& record, const RecordPositions* pos);
    bool StoreRecordUnsafe(const DbRecord& record);
    bool UpdateRecordUnsafe(DbRecord& record);
    bool RemoveRecordUnsafe(const std::string& key);

    template <class T>
    inline boost::optional<DbRecord> FindRecordUnsafe(const T& problem_config)
    {
        std::string key = DbRecord::Serialize(problem_config);
        return FindRecordUnsafe(key, nullptr);
    }
};
} // namespace miopen

#endif // GUARD_MIOPEN_DB_HPP_
