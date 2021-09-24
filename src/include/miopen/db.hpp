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
#ifndef GUARD_MIOPEN_DB_HPP_
#define GUARD_MIOPEN_DB_HPP_

#include <miopen/db_record.hpp>
#include <miopen/rank.hpp>

#include <boost/core/explicit_operator_bool.hpp>
#include <boost/none.hpp>
#include <boost/optional/optional.hpp>

#include <chrono>
#include <string>

namespace boost {
namespace filesystem {
class path;
} // namespace filesystem
} // namespace boost

namespace miopen {

struct RecordPositions
{
    std::streamoff begin = -1;
    std::streamoff end   = -1;
};

class LockFile;

constexpr bool DisableUserDbFileIO = MIOPEN_DISABLE_USERDB;

/// No instance of this class should be used from several threads at the same time.
class PlainTextDb
{
    public:
    PlainTextDb(const std::string& filename_, bool is_system = false);

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

    protected:
    LockFile& GetLockFile() { return lock_file; }
    const std::string& GetFileName() const { return filename; }
    bool IsWarningIfUnreadable() const { return warning_if_unreadable; }
    boost::optional<DbRecord> FindRecordUnsafe(const std::string& key, RecordPositions* pos);
    bool StoreRecordUnsafe(const DbRecord& record);
    bool UpdateRecordUnsafe(DbRecord& record);
    bool RemoveRecordUnsafe(const std::string& key);

    private:
    std::string filename;
    LockFile& lock_file;
    const bool warning_if_unreadable;

    bool FlushUnsafe(const DbRecord& record, const RecordPositions* pos);

    template <class T>
    inline boost::optional<DbRecord> FindRecordUnsafe(const T& problem_config)
    {
        const auto key = DbRecord::Serialize(problem_config);
        return FindRecordUnsafe(key, nullptr);
    }
};

template <class TDb, class TRet = decltype(TDb::GetCached("", true))>
TRet GetDbInstance(rank<1>, const std::string& path, bool is_system)
{
    return TDb::GetCached(path, is_system);
};

template <class TDb>
TDb GetDbInstance(rank<0>, const std::string& path, bool is_system)
{
    return {path, is_system};
};

template <class TDb, class TRet = decltype(GetDbInstance<TDb>(rank<1>{}, {}, {}))>
TRet GetDbInstance(const std::string& path, bool is_system = true)
{
    return GetDbInstance<TDb>(rank<1>{}, path, is_system);
}

template <class TInstalled, class TUser, bool merge_records>
class MultiFileDb
{
    public:
    MultiFileDb(const std::string& installed_path, const std::string& user_path)
        : _installed(GetDbInstance<TInstalled>(installed_path, true))
#if !MIOPEN_DISABLE_USERDB
          ,
          _user(GetDbInstance<TUser>(user_path, false))
#endif
    {
    }

    template <bool merge = merge_records, std::enable_if_t<merge>* = nullptr, typename... U>
    auto FindRecord(const U&... args)
    {
        auto users     = _user.FindRecord(args...);
        auto installed = _installed.FindRecord(args...);

        if(users && installed)
        {
            users->Merge(installed.value());
            return users;
        }

        if(users)
            return users;

        return installed;
    }

    template <bool merge = merge_records, std::enable_if_t<!merge>* = nullptr, typename... U>
    auto FindRecord(const U&... args)
    {
        auto users = _user.FindRecord(args...);
        return users ? users : _installed.FindRecord(args...);
    }

    template <typename... U>
    auto StoreRecord(const U&... args)
    {
        return _user.StoreRecord(args...);
    }

    template <typename... U>
    auto UpdateRecord(U&... args)
    {
        return _user.UpdateRecord(args...);
    }

    template <typename... U>
    auto RemoveRecord(const U&... args)
    {
        return _user.RemoveRecord(args...);
    }

    template <typename... U>
    auto Update(const U&... args)
    {
        return _user.Update(args...);
    }

    template <typename... U>
    auto Load(U&... args)
    {
        if(_user.Load(args...))
            return true;
        return _installed.Load(args...);
    }

    template <typename... U>
    auto Remove(const U&... args)
    {
        return _user.Remove(args...);
    }

    private:
    template <class TDb, class TRet = decltype(TDb::GetCached("", true))>
    static TRet GetDbInstance(rank<1>, const std::string& path, bool warn_if_unreadable)
    {
        return TDb::GetCached(path, warn_if_unreadable);
    };

    template <class TDb>
    static TDb GetDbInstance(rank<0>, const std::string& path, bool warn_if_unreadable)
    {
        return {path, warn_if_unreadable};
    };

    template <class TDb, class TRet = decltype(GetDbInstance<TDb>(rank<1>{}, {}, {}))>
    static TRet GetDbInstance(const std::string& path, bool warn_if_unreadable)
    {
        return GetDbInstance<TDb>(rank<1>{}, path, warn_if_unreadable);
    }

    decltype(MultiFileDb::GetDbInstance<TInstalled>("", true)) _installed;
#if !MIOPEN_DISABLE_USERDB
    decltype(MultiFileDb::GetDbInstance<TUser>("", false)) _user;
#endif
};

template <class TInnerDb>
class DbTimer
{
    public:
    template <class... TArgs>
    DbTimer(TArgs&&... args) : inner(args...)
    {
    }

    template <typename... U>
    auto FindRecord(const U&... args)
    {
        return Measure("FindRecord", [&]() { return inner.FindRecord(args...); });
    }

    template <typename... U>
    auto StoreRecord(U&... record)
    {
        return Measure("StoreRecord", [&]() { return inner.StoreRecord(record...); });
    }

    template <typename... U>
    auto UpdateRecord(U&... args)
    {
        return Measure("UpdateRecord", [&]() { return inner.UpdateRecord(args...); });
    }

    template <typename... U>
    auto RemoveRecord(const U&... args)
    {
        return Measure("RemoveRecord", [&]() { return inner.RemoveRecord(args...); });
    }

    template <typename... U>
    auto Update(const U&... args)
    {
        return Measure("Update", [&]() { return inner.Update(args...); });
    }

    template <typename... U>
    bool Load(U&... args)
    {
        return Measure("Load", [&]() { return inner.Load(args...); });
    }

    template <typename... U>
    bool Remove(const U&... args)
    {
        return Measure("Remove", [&]() { return inner.Remove(args...); });
    }

    private:
    TInnerDb inner;

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
};
} // namespace miopen

#endif // GUARD_MIOPEN_DB_HPP_
