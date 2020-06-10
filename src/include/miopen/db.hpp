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

struct RecordPositions;
class LockFile;

/// No instance of this class should be used from several threads at the same time.
class PlainTextDb
{
    public:
    PlainTextDb(const std::string& filename_,
                bool is_system,
                const std::string& arch,
                std::size_t num_cu);

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

#if MIOPEN_DISABLE_USERDB
struct sink
{
    template <typename... Args>
    sink(Args const&...)
    {
    }
};
#endif

template <class TInstalled, class TUser, bool merge_records>
class MultiFileDb
{
    public:
    MultiFileDb(const std::string& installed_path,
                const std::string& user_path,
                const std::string& arch  = "",
                const std::size_t num_cu = 0)
        : _installed(GetDbInstance<TInstalled>(installed_path, true, arch, num_cu))
#if !MIOPEN_DISABLE_USERDB
          ,
          _user(GetDbInstance<TUser>(user_path, false, arch, num_cu))
#endif
    {
#if MIOPEN_DISABLE_USERDB
        (void)(user_path);
#endif
    }

    template <bool merge = merge_records, std::enable_if_t<merge>* = nullptr, typename... U>
    auto FindRecord(const U&... args)
    {
#if !MIOPEN_DISABLE_USERDB
        auto users = _user.FindRecord(args...);
#endif
        auto installed = _installed.FindRecord(args...);

#if !MIOPEN_DISABLE_USERDB
        if(users && installed)
        {
            users->Merge(installed.value());
            return users;
        }

        if(users)
            return users;
#endif

        return installed;
    }

    template <bool merge = merge_records, std::enable_if_t<!merge>* = nullptr, typename... U>
    auto FindRecord(const U&... args)
    {
#if !MIOPEN_DISABLE_USERDB
        auto users = _user.FindRecord(args...);
        return users ? users : _installed.FindRecord(args...);
#else
        return _installed.FindRecord(args...);
#endif
    }

    template <typename... U>
    auto StoreRecord(const U&... args)
    {
#if MIOPEN_DISABLE_USERDB
        sink{args...};
        return true;
#else
        return _user.StoreRecord(args...);
#endif
    }

    template <typename... U>
    auto UpdateRecord(U&... args)
    {
#if MIOPEN_DISABLE_USERDB
        sink{args...};
        return true;
#else
        return _user.UpdateRecord(args...);
#endif
    }

    template <typename... U>
    auto RemoveRecord(const U&... args)
    {
#if MIOPEN_DISABLE_USERDB
        sink{args...};
        return true;
#else
        return _user.RemoveRecord(args...);
#endif
    }

    template <typename... U>
    auto Update(const U&... args)
    {
#if MIOPEN_DISABLE_USERDB
        sink{args...};
        return true;
#else
        return _user.Update(args...);
#endif
    }

    template <typename... U>
    auto Load(U&... args)
    {
#if !MIOPEN_DISABLE_USERDB
        if(_user.Load(args...))
            return true;
#endif
        return _installed.Load(args...);
    }

    template <typename... U>
    auto Remove(const U&... args)
    {
#if MIOPEN_DISABLE_USERDB
        sink{args...};
        return true;
#else
        return _user.Remove(args...);
#endif
    }

    private:
    template <class TDb, class TRet = decltype(TDb::GetCached("", true, "", 0))>
    static TRet GetDbInstance(rank<1>,
                              const std::string& path,
                              bool warn_if_unreadable,
                              const std::string& arch,
                              std::size_t num_cu)
    {
        return TDb::GetCached(path, warn_if_unreadable, arch, num_cu);
    };

    template <class TDb>
    static TDb GetDbInstance(rank<0>,
                             const std::string& path,
                             bool warn_if_unreadable,
                             const std::string& arch,
                             std::size_t num_cu)
    {
        return {path, warn_if_unreadable, arch, num_cu};
    };

    template <class TDb, class TRet = decltype(GetDbInstance<TDb>(rank<1>{}, {}, {}, {}, {}))>
    static TRet GetDbInstance(const std::string& path,
                              bool warn_if_unreadable,
                              const std::string& arch,
                              const std::size_t num_cu)
    {
        return GetDbInstance<TDb>(rank<1>{}, path, warn_if_unreadable, arch, num_cu);
    }

    decltype(MultiFileDb::GetDbInstance<TInstalled>("", true, "", 0)) _installed;
#if !MIOPEN_DISABLE_USERDB
    decltype(MultiFileDb::GetDbInstance<TUser>("", false, "", 0)) _user;
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
