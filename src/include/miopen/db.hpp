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

#include <miopen/db_record.hpp>

#include <boost/core/explicit_operator_bool.hpp>
#include <boost/none.hpp>
#include <boost/optional/optional.hpp>

#include <string>

namespace boost {
namespace filesystem {
class path;
} // namespace filesystem
} // namespace boost

namespace miopen {

struct RecordPositions;
class LockFile;

std::string LockFilePath(const boost::filesystem::path& filename_);

/// No instance of this class should be used from several threads at the same time.
class Db
{
    public:
    Db(const std::string& filename_, bool is_system = true);

    /// Searches db for provided key and returns found record or none if key not found in database
    boost::optional<DbRecord> FindRecord(const std::string& key);

    template <class T>
    inline boost::optional<DbRecord> FindRecord(const T& problem_config)
    {
        const auto key = DbRecord::Serialize(problem_config);
        return FindRecord(key);
    }

    /// Stores provided record in database. If record with same key is already in database it is
    /// replaced by provided record.
    ///
    /// Returns true if store was successful, false otherwise.
    bool StoreRecord(const DbRecord& record);

    /// Stores provided record in database. If record with same key is already in database it is
    /// updated with values from provided record. Provided records data is also updated via
    /// DbRecord::Merge().
    ///
    /// Returns true if update was successful, false otherwise.
    bool UpdateRecord(DbRecord& record);

    /// Removes record with provided key from db
    ///
    /// Returns true if remove was successful, false otherwise.
    bool RemoveRecord(const std::string& key);

    /// Removes ID with associated VALUES from record with key PROBLEM_CONFIG from db.
    /// If payload of a record becomes empty after that, also removes the entire record
    ///
    /// Returns true if remove was successful. Returns false if this PROBLEM_CONFIG or ID was not
    /// found.
    template <class T>
    inline bool Remove(const T& problem_config, const std::string& id)
    {
        const auto key = DbRecord::Serialize(problem_config);
        return Remove(key, id);
    }

    bool Remove(const std::string& key, const std::string& id);

    template <class T>
    inline bool RemoveRecord(const T& problem_config)
    {
        const auto key = DbRecord::Serialize(problem_config);
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
        const auto ok = UpdateRecord(record);
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
        const auto record = FindRecord(problem_config);

        if(!record)
            return false;
        return record->GetValues(id, values);
    }

    private:
    std::string filename;
    LockFile& lock_file;
    const bool warn_if_unreadable;

    boost::optional<DbRecord> FindRecordUnsafe(const std::string& key, RecordPositions* pos);
    bool FlushUnsafe(const DbRecord& record, const RecordPositions* pos);
    bool StoreRecordUnsafe(const DbRecord& record);
    bool UpdateRecordUnsafe(DbRecord& record);
    bool RemoveRecordUnsafe(const std::string& key);

    template <class T>
    inline boost::optional<DbRecord> FindRecordUnsafe(const T& problem_config)
    {
        const auto key = DbRecord::Serialize(problem_config);
        return FindRecordUnsafe(key, nullptr);
    }
};

class MultiFileDb
{
    public:
    MultiFileDb(const std::string& installed_path, const std::string& user_path)
        : _installed(installed_path), _user(user_path, false)
    {
    }

    boost::optional<DbRecord> FindRecord(const std::string& key)
    {
        auto users           = _user.FindRecord(key);
        const auto installed = _installed.FindRecord(key);

        if(users && installed)
        {
            users->Merge(installed.value());
            return users;
        }

        if(users)
            return users;

        return installed;
    }

    template <class T>
    boost::optional<DbRecord> FindRecord(const T& problem_config)
    {
        auto users           = _user.FindRecord(problem_config);
        const auto installed = _installed.FindRecord(problem_config);

        if(users && installed)
        {
            users->Merge(installed.value());
            return users;
        }

        if(users)
            return users;

        return installed;
    }

    bool StoreRecord(const DbRecord& record) { return _user.StoreRecord(record); }

    bool UpdateRecord(DbRecord& record) { return _user.UpdateRecord(record); }

    bool RemoveRecord(const std::string& key) { return _user.RemoveRecord(key); }

    template <class T>
    bool RemoveRecord(const T& problem_config)
    {
        return _user.RemoveRecord(problem_config);
    }

    template <class T, class V>
    boost::optional<DbRecord>
    Update(const T& problem_config, const std::string& id, const V& values)
    {
        return _user.Update(problem_config, id, values);
    }

    template <class T, class V>
    bool Load(const T& problem_config, const std::string& id, V& values)
    {
        if(_user.Load(problem_config, id, values))
            return true;

        return _installed.Load(problem_config, id, values);
    }

    template <class T>
    bool Remove(const T& problem_config, const std::string& id)
    {
        return _user.Remove(problem_config, id);
    }

    private:
    Db _installed, _user;
};
} // namespace miopen

#endif // GUARD_MIOPEN_DB_HPP_
