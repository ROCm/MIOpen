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
#ifndef GUARD_MIOPEN_DB_RECORD_HPP_
#define GUARD_MIOPEN_DB_RECORD_HPP_

#include <miopen/config.h>
#include <miopen/logger.hpp>

#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/interprocess/sync/named_recursive_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/optional.hpp>

#include <atomic>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace miopen {

    class LockFile
    {
        friend class DbLockFileDispatcher;

    private:
        class Impl
        {
            Impl(const Impl&) = delete;
            Impl operator=(const Impl&) = delete;

        public:
            Impl(const char* path) : _file(path), _file_lock(path) {}

            ~Impl()
            {
                if (_locked)
                    unlock();
            }

            void lock()
            {
                _mutex.lock();
                _locked = true;
                _file_lock.lock();
            }

            void unlock() 
            {
                _file_lock.unlock();
                _locked = false;
                _mutex.unlock();
            }

        private:
            bool _locked;
            std::mutex _mutex;
            std::ofstream _file;
            boost::interprocess::file_lock _file_lock;
        };

    public:
        LockFile(Impl& impl) : _impl(impl) {}
        void lock() { _impl.lock(); }
        void unlock() { _impl.unlock(); }

    private:
        Impl& _impl;
    };

    class DbLockFileDispatcher
    {
        DbLockFileDispatcher() = delete;

    public:
        static LockFile Get(const char* path)
        {
            { // To guarantee that construction won't be called if not required.
                auto found = LockFiles().find(path);

                if (found != LockFiles().end())
                    return found->second;
            }

            auto emplaced = LockFiles().emplace(std::piecewise_construct, std::forward_as_tuple(path), std::forward_as_tuple(path));
            return { emplaced.first->second };
        }

    private:
        static std::map<std::string, LockFile::Impl>& LockFiles()
        {
            static std::map<std::string, LockFile::Impl> lock_files;
            return lock_files;
        }
    };

    template<class TBase>
    class RecursiveBasicLocakable
    {
    public:
        template<class... TArgs>
        RecursiveBasicLocakable(TArgs... args) : _base(args...) {}

        void lock()
        {
            if (_locks++ > 0)
                return;
            _base.lock();
        }

        void unlock()
        {
            assert(_locks > 0);
            if (--_locks > 0)
                return;
            _base.unlock();
        }

    private:
        TBase _base;
        unsigned _locks = 0;
    };

class Db;

/// db consists of 0 or more records.
/// Each record is an ASCII text line.
/// Record format:
///   [ KEY "=" ID ":" VALUES { ";" ID ":" VALUES} ]
///
/// KEY - An identifer of a record.
/// ID - Can be considered as a sub-key under which respective VALUES are stored.
/// VALUES - A data associated with specific ID under the KEY. Intended to represent a set of
/// values, hence the name.
///
/// Neither of ";:=" within KEY, ID and VALUES is allowed.
/// There should be none identical KEYs in the same db file.
/// There should be none identical IDs within the same record.
///
/// Intended usage:
/// KEY: A stringized problem config.
/// ID: A symbolic name of the Solver applicable for the KEY (problem config). There could be
/// several Solvers able to handle the same config, so several IDs can be put under a KEY.
/// Format of VALUES stored under each ID is Solver-specific. in other words, how a set of values
/// (or whatever a Solver wants to store in VALUES) is encoded into a string depends on the Solver.
/// Note: If VALUES is used to represent a set of numeric values, then it is recommended to use ","
/// as a separator.

/// Represents a db record associated with specific KEY.
/// Ctor arguments are path to db file and a KEY (or an object able to provide a KEY).
/// Upon construction, allows getting and modifying contents of a record (IDs and VALUES).
///
/// \todo Separate "db file" and "db record" abstractions.
/// \todo The Store() operation is neither MP- nor MT-safe.
class DbRecord
{
    private:
    std::string key;
    std::unordered_map<std::string, std::string> map;

    template <class T>
    static // 'static' is for calling from ctor
        std::string
        Serialize(const T& data)
    {
        std::ostringstream ss;
        data.Serialize(ss);
        return ss.str();
    }

    bool ParseContents(const std::string& contents);
    void WriteContents(std::ostream& stream) const;
    bool SetValues(const std::string& id, const std::string& values);
    bool GetValues(const std::string& id, std::string& values) const;

    DbRecord(const std::string& key_) : key(key_) {}

    public:
    /// T shall provide a db KEY by means of the "void Serialize(std::ostream&) const" member
    /// function.
    template <class T>
    DbRecord(const T& problem_config_) : DbRecord(Serialize(problem_config_))
    {
    }

    /// Merges data from this record to data from that record if their keys are same.
    /// This record would contain all ID:VALUES pairs from that record that are not in this.
    /// E.g. this = {ID1:VALUE1}
    ///      that = {ID1:VALUE3, ID2:VALUE2}
    ///      this.Merge(that) = {ID1:VALUE1, ID2:VALUE2}
    void Merge(const DbRecord& that);

    /// Obtains VALUES from an object of class T and sets it in record (in association with ID,
    /// under the current KEY).
    /// T shall have the "void Serialize(std::ostream&) const" member function available.
    ///
    /// Returns true if records data was changed.
    template <class T>
    bool SetValues(const std::string& id, const T& values)
    {
        return SetValues(id, Serialize(values));
    }

    /// Get VALUES associated with ID under the current KEY and delivers those to a member function
    /// of a class T object. T shall have the "bool Deserialize(const std::string& str)"
    /// member function available.
    ///
    /// Returns false if there is none ID:VALUES in the record or in case of any error, e.g. if
    /// VALUES cannot be deserialized due to incorrect format.
    template <class T>
    bool GetValues(const std::string& id, T& values) const
    {
        std::string s;
        if(!GetValues(id, s))
            return false;

        const bool ok = values.Deserialize(s);
        if(!ok)
            MIOPEN_LOG(LoggingLevel::Error, "deserialize failed: " << s);
        return ok;
    }

    /// Removes ID with associated VALUES from this record.
    ///
    /// Returns true if erase was successful. Returns false if this ID was not found.
    bool EraseValues(const std::string& id);

    friend class Db;
};

/// No instance of this class should be used from several threads at the same time.
class Db
{
    private:
    struct RecordPositions
    {
        std::streamoff begin = -1;
        std::streamoff end   = -1;
    };

    using exclusive_lock =
        boost::interprocess::scoped_lock<RecursiveBasicLocakable<LockFile>>;

    std::string filename;
    RecursiveBasicLocakable<LockFile> lock_file;

    boost::optional<DbRecord> FindRecord(const std::string& key, RecordPositions* pos);
    bool Flush(const DbRecord& record, const RecordPositions* pos);

    public:
    Db(const std::string& filename_) : filename(filename_), lock_file(DbLockFileDispatcher::Get((filename_ + ".lock").c_str())) {}

    /// Searches db for provided key and returns found reconrd or none if key not found in database
    boost::optional<DbRecord> FindRecord(const std::string& key)
    {
        return FindRecord(key, nullptr);
    }

    template <class T>
    boost::optional<DbRecord> FindRecord(const T& problem_config)
    {
        std::string key = DbRecord::Serialize(problem_config);
        return FindRecord(key, nullptr);
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

    template <class T>
    bool RemoveRecord(const T& problem_config)
    {
        std::string key = DbRecord::Serialize(problem_config);
        return RemoveRecord(key);
    }

    /// Updates record under key PROBLEM_CONFIG  with data ID:VALUES in database.
    /// Both T and V classes should have "void Serialize(std::ostream&) const" member function
    /// available.
    ///
    /// Returns updated record or none if update was unsuccessful.
    template <class T, class V>
    boost::optional<DbRecord>
    Store(const T& problem_config, const std::string& id, const V& values)
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
    bool Load(const T& problem_config, const std::string& id, V& values)
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
    bool Remove(const T& problem_config, const std::string& id)
    {
        exclusive_lock lock(lock_file);
        auto record = FindRecord(problem_config);
        if(!record)
            return false;
        bool erased = record->EraseValues(id);
        if(!erased)
            return false;
        return StoreRecord(*record);
    }
};
} // namespace miopen

#endif // GUARD_MIOPEN_DB_RECORD_HPP_
