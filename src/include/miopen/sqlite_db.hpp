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

#include <miopen/db_record.hpp>
#include <miopen/manage_ptr.hpp>
#include <miopen/errors.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/lock_file.hpp>

#include <boost/core/explicit_operator_bool.hpp>
#include <boost/none.hpp>
#include <boost/optional/optional.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread_time.hpp>
#include "sqlite3.h"
#include <mutex>
#include <thread>

#include <string>
#include <chrono>
#include <unordered_map>

namespace boost {
namespace filesystem {
class path;
} // namespace filesystem
} // namespace boost

namespace miopen {

#define MIOPEN_VALIDATE_LOCK(lock)                       \
    do                                                   \
    {                                                    \
        if(!(lock))                                      \
            MIOPEN_THROW("Db lock has failed to lock."); \
    } while(false)

template <class Derived>
struct SQLiteSerializable
{
    std::tuple<std::string, std::vector<std::string>> WhereClause() const
    {
        std::vector<std::string> values;
        std::vector<std::string> clauses;
        Derived::Visit(static_cast<const Derived&>(*this),
                       [&](const std::string& value, const std::string& name) {
                           clauses.push_back("(" + name + " = ? )");
                           values.push_back(value);
                       });
        Derived::Visit(static_cast<const Derived&>(*this),
                       [&](const int value, const std::string name) {
                           clauses.push_back("(" + name + " = ? )");
                           values.push_back(std::to_string(value));
                       });
        std::string clause = JoinStrings(clauses, " AND ");
        return std::make_tuple(clause, values);
    }
    std::tuple<std::string, std::vector<std::string>> InsertQuery() const
    {
        std::vector<std::string> int_names, str_names, values;
        Derived::Visit(static_cast<const Derived&>(*this),
                       [&](const std::string& value, const std::string& name) {
                           str_names.push_back(name);
                           values.push_back(value);
                       });
        Derived::Visit(static_cast<const Derived&>(*this),
                       [&](const int value, const std::string name) {
                           int_names.push_back(name);
                           values.push_back(std::to_string(value));
                       });
        std::vector<std::string> tokens((values.size()), "?");
        ;

        std::string q = "INSERT OR IGNORE INTO " + Derived::table_name() + "( " +
                        JoinStrings(str_names, ",") + "," + JoinStrings(int_names, ",") +
                        " ) VALUES( " + JoinStrings(tokens, ",") + ");";
        return std::make_tuple(q, values);
    }
    std::tuple<std::string, std::vector<std::string>> SelectQuery() const
    {
        std::string clauses;
        std::vector<std::string> values;
        std::tie(clauses, values) = WhereClause();
        std::string query = "SELECT id FROM " + Derived::table_name() + " WHERE " + clauses + ";";
        return std::make_tuple(query, values);
    }

    std::string CreateQuery() const
    {
        std::vector<std::string> str_fields;
        Derived::Visit(static_cast<const Derived&>(*this),
                       [&](const std::string value, const std::string name) {
                           std::ignore = value;
                           str_fields.push_back(name);
                       });
        std::vector<std::string> int_fields;
        Derived::Visit(static_cast<const Derived&>(*this),
                       [&](const int value, const std::string name) {
                           std::ignore = value;
                           int_fields.push_back(name);
                       });
        std::ostringstream ss;
        ss << "CREATE TABLE IF NOT EXISTS `" << Derived::table_name() << "` ("
           << "`id` INTEGER PRIMARY KEY ASC";
        for(auto& el : str_fields)
            ss << ",`" << el << "` TEXT NOT NULL";
        for(auto& el : int_fields)
            ss << ",`" << el << "` INT NOT NULL";
        ss << ");";
        ss << "CREATE UNIQUE INDEX IF NOT EXISTS "
           << "`idx_" << Derived::table_name() << "` "
           << "ON " << Derived::table_name() << "( " << miopen::JoinStrings(str_fields, ",") << ", "
           << miopen::JoinStrings(int_fields, ",") << " );";
        return ss.str();
    }
};

template <typename Derived>
class SQLiteBase
{
    protected:
    using sqlite3_ptr      = MIOPEN_MANAGE_PTR(sqlite3*, sqlite3_close);
    using exclusive_lock   = boost::unique_lock<LockFile>;
    using shared_lock      = boost::shared_lock<LockFile>;
    using sqlite3_stmt_ptr = MIOPEN_MANAGE_PTR(sqlite3_stmt*, sqlite3_finalize);
    static boost::system_time GetLockTimeout()
    {
        return boost::get_system_time() + boost::posix_time::milliseconds(60000);
    }

    public:
    SQLiteBase(const std::string& filename_,
               bool is_system,
               const std::string& arch_,
               std::size_t num_cu_)
        : filename(filename_),
          arch(arch_),
          num_cu(num_cu_),
          lock_file(LockFile::Get(LockFilePath(filename_).c_str()))
    {
        MIOPEN_LOG_I2("Initializing " << (is_system ? "system" : "user") << " database file "
                                      << filename);
        if(!is_system && !filename.empty())
        {
            auto file            = boost::filesystem::path(filename_);
            const auto directory = file.remove_filename();

            if(!(boost::filesystem::exists(directory)))
            {
                if(!boost::filesystem::create_directories(directory))
                    MIOPEN_LOG_W("Unable to create a directory: " << directory);
                else
                    boost::filesystem::permissions(directory, boost::filesystem::all_all);
            }
        }
        sqlite3* ptr_tmp;
        int rc = 0;
        if(is_system)
            rc = sqlite3_open_v2(filename_.c_str(), &ptr_tmp, SQLITE_OPEN_READONLY, nullptr);
        else
            rc = sqlite3_open_v2(
                filename_.c_str(), &ptr_tmp, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
        ptrDb = sqlite3_ptr{ptr_tmp};
        if(rc != 0)
        {
            dbInvalid = true;
            if(!is_system)
            {
                MIOPEN_THROW(miopenStatusInternalError, "Cannot open database file:" + filename_);
            }
            else
            {
                MIOPEN_LOG_W("Unable to read system database file:" + filename_ +
                             " Performance may degrade");
            }
        }
        else
            dbInvalid = false;
    }

    static Derived&
    GetCached(const std::string& path, bool is_system, const std::string& arch, std::size_t num_cu);
    // TODO: Fix this for the overhead of having fields per record
    using SQLRes_t = std::vector<std::unordered_map<std::string, std::string>>;

    static int find_callback(void* _res, int argc, char** argv, char** azColName)
    {
        SQLRes_t* res = static_cast<SQLRes_t*>(_res);
        std::unordered_map<std::string, std::string> record;
        for(auto i               = 0; i < argc; i++)
            record[azColName[i]] = (argv[i] != nullptr) ? argv[i] : "NULL";
        res->push_back(record);
        return 0;
    }

    inline auto SQLExec(const std::string& query)
    {
        MIOPEN_LOG_T(std::this_thread::get_id() << ":" << query);
        {
            auto rc = SQLRety([&]() {
                return sqlite3_exec(ptrDb.get(), query.c_str(), find_callback, nullptr, nullptr);
            });
            if(rc != SQLITE_OK)
            {
                MIOPEN_LOG_I2(query);
                MIOPEN_THROW(miopenStatusInternalError, SQLErrorMessage());
                sqlite3_close(ptrDb.get());
                return false;
            }
        }
        return true;
    }
    inline auto SQLExec(const std::string& query, SQLRes_t& res) const
    {
        res.clear();
        MIOPEN_LOG_T(std::this_thread::get_id() << ":" << query);
        {
            auto rc = SQLRety([&]() {
                return sqlite3_exec(
                    ptrDb.get(), query.c_str(), find_callback, static_cast<void*>(&res), nullptr);
            });
            if(rc != SQLITE_OK)
            {
                MIOPEN_LOG_I2(query);
                MIOPEN_THROW(miopenStatusInternalError, SQLErrorMessage());
                sqlite3_close(ptrDb.get());
                return false;
            }
        }
        return true;
    }

    template <class F>
    inline int SQLRety(F f) const
    {
        auto timeout_end = std::chrono::high_resolution_clock::now() +
                           std::chrono::seconds(30); // TODO: make configurable
        auto tries = 0;
        while(true)
        {
            int rc = f();
            if(rc == SQLITE_BUSY)
            {
                MIOPEN_LOG_I2("Database" + filename + "  busy, retrying ...");
                ++tries;
                if(tries > 50)
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                else
                    std::this_thread::yield();
            }
            else
                return rc;
            if(std::chrono::high_resolution_clock::now() > timeout_end)
                MIOPEN_THROW("Timeout while waiting for Database: " + filename);
        }
    }

    inline std::string SQLErrorMessage() const
    {
        std::string errMsg = "Internal error while accessing SQLite database: ";
        return errMsg + sqlite3_errmsg(ptrDb.get());
    }

    auto Prepare(const std::string& query) const
    {
        sqlite3_stmt* ptr = nullptr;
        MIOPEN_LOG_I2(query);
        auto rc = sqlite3_prepare_v2(ptrDb.get(), query.c_str(), query.size(), &ptr, nullptr);
        if(rc != SQLITE_OK)
        {
            std::string err_msg = "SQLite prepare error: ";
            MIOPEN_THROW(miopenStatusInternalError, err_msg + sqlite3_errmsg(ptrDb.get()));
        }
        return sqlite3_stmt_ptr{ptr};
    }
    auto PrepareAndBind(const std::string& query, std::vector<std::string>& values) const
    {
        auto stmt = Prepare(query);
        int cnt   = 1;
        for(auto& kinder : values)
        {
            auto rc = sqlite3_bind_text(
                stmt.get(), cnt++, kinder.data(), kinder.size(), SQLITE_TRANSIENT); // NOLINT
            if(rc != SQLITE_OK)
                MIOPEN_THROW(miopenStatusInternalError, SQLErrorMessage());
        }
        MIOPEN_LOG_I2("[" << JoinStrings(values, ",") << "]");
        return stmt;
    }

    template <typename... U>
    inline auto FindRecord(U&... args)
    {
        const auto lock = shared_lock(lock_file, GetLockTimeout());
        MIOPEN_VALIDATE_LOCK(lock);
        return reinterpret_cast<Derived*>(this)->FindRecordUnsafe(args...);
    }

    template <typename... U>
    inline auto RemoveRecord(U&... args)
    {
        const auto lock = exclusive_lock(lock_file, GetLockTimeout());
        MIOPEN_VALIDATE_LOCK(lock);
        return reinterpret_cast<Derived*>(this)->RemoveRecordUnsafe(args...);
    }

    template <typename... U>
    inline auto StoreRecord(U&... args)
    {
        const auto lock = exclusive_lock(lock_file, GetLockTimeout());
        MIOPEN_VALIDATE_LOCK(lock);
        return reinterpret_cast<Derived*>(this)->StoreRecordUnsafe(args...);
    }

    template <typename... U>
    inline auto Remove(const U&... args)
    {
        const auto lock = exclusive_lock(lock_file, GetLockTimeout());
        MIOPEN_VALIDATE_LOCK(lock);
        return reinterpret_cast<Derived*>(this)->RemoveUnsafe(args...);
    }

    template <typename... U>
    inline auto Update(const U&... args)
    {
        const auto lock = exclusive_lock(lock_file, GetLockTimeout());
        MIOPEN_VALIDATE_LOCK(lock);
        return reinterpret_cast<Derived*>(this)->UpdateUnsafe(args...);
    }

    template <typename... U>
    inline auto Load(U&&... args)
    {
        const auto lock = shared_lock(lock_file, GetLockTimeout());
        MIOPEN_VALIDATE_LOCK(lock);
        return reinterpret_cast<Derived*>(this)->LoadUnsafe(args...);
    }

    protected:
    std::string filename;
    std::string arch;
    size_t num_cu;
    LockFile& lock_file;
    sqlite3_ptr ptrDb = nullptr;
    bool dbInvalid;
};

template <typename Derived>
Derived& SQLiteBase<Derived>::GetCached(const std::string& path,
                                        bool is_system,
                                        const std::string& arch,
                                        const size_t num_cu)
{
    static std::mutex mutex;
    static const std::lock_guard<std::mutex> lock{mutex};

    static auto instances = std::map<std::string, Derived*>{};
    const auto it         = instances.find(path);

    if(it != instances.end())
        return *(it->second);

    instances.emplace(path, new Derived{path, is_system, arch, num_cu}); // NOLINT
    return *(instances.at(path));
}

class SQLitePerfDb : public SQLiteBase<SQLitePerfDb>
{
    public:
    SQLitePerfDb(const std::string& filename_,
                 bool is_system,
                 const std::string& arch_,
                 std::size_t num_cu_);

    template <class T>
    inline void InsertConfig(const T& prob_desc)
    {
        std::string clause;
        std::vector<std::string> vals;
        std::tie(clause, vals) = prob_desc.InsertQuery();
        auto stmt = PrepareAndBind(clause, vals);
        auto rc   = SQLRety([&]() { return sqlite3_step(stmt.get()); });
        if(rc != SQLITE_DONE)
            MIOPEN_THROW(miopenStatusInternalError,
                         "Failed to insert config: " + SQLErrorMessage());
        auto cnt = sqlite3_changes(ptrDb.get());
        MIOPEN_LOG_I2(cnt << " rows updated");
    }
    template <class T>
    inline std::string GetConfigIDs(const T& prob_desc)
    {
        std::string clause;
        std::vector<std::string> vals;
        std::tie(clause, vals) = prob_desc.WhereClause();
        auto query = "SELECT id FROM " + prob_desc.table_name() + " WHERE ( " + clause + " );";
        auto stmt  = PrepareAndBind(query, vals);
        while(true)
        {
            auto rc = SQLRety([&]() { return sqlite3_step(stmt.get()); });
            if(rc == SQLITE_ROW)
            {
                auto id =
                    std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt.get(), 0)));
                return id;
            }
            else if(rc == SQLITE_DONE)
                return "";
            else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
                MIOPEN_THROW(miopenStatusInternalError, SQLErrorMessage());
        }
    }
    template <typename T>
    inline boost::optional<DbRecord> FindRecordUnsafe(const T& problem_config)
    {
        if(dbInvalid)
            return boost::none;
        std::string clause;
        std::vector<std::string> values;
        std::tie(clause, values) = problem_config.WhereClause();
        // clang-format off
        auto select_query =
            "SELECT solver, params "
            "FROM perf_db "
            "INNER JOIN " + problem_config.table_name() + " " 
            "ON perf_db.config = " + problem_config.table_name() +".id "
            "WHERE "
            "( " + clause + " )"
            "AND (arch = '" + arch + "' ) "
            "AND (num_cu = '" + std::to_string(num_cu) + "');";
        // clang-format on
        auto stmt = PrepareAndBind(select_query, values);
        DbRecord rec;
        while(true)
        {
            auto rc = SQLRety([&]() { return sqlite3_step(stmt.get()); });
            if(rc == SQLITE_ROW)
            {
                auto c_slvr   = sqlite3_column_text(stmt.get(), 0);
                auto c_params = sqlite3_column_text(stmt.get(), 1);
                rec.SetValues(std::string(reinterpret_cast<const char*>(c_slvr)),
                              std::string(reinterpret_cast<const char*>(c_params)));
            }
            else if(rc == SQLITE_DONE)
                break;
            else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
                MIOPEN_THROW(miopenStatusInternalError, SQLErrorMessage());
        }
        if(rec.GetSize() == 0)
            return boost::none;
        else
            return boost::optional<DbRecord>(rec);
    }

    /// Removes ID with associated VALUES from record with key PROBLEM_CONFIG from db.
    ///
    /// Returns true if remove was successful. Returns false if this PROBLEM_CONFIG or ID was not
    /// found.
    template <class T>
    inline bool RemoveUnsafe(const T& problem_config, const std::string& id)
    {
        if(dbInvalid)
            return false;
        std::string clause;
        std::vector<std::string> values;
        std::tie(clause, values) = problem_config.WhereClause();
        // clang-format off
        auto query = 
            "DELETE FROM perf_db "
            "WHERE config IN ("
            "SELECT id FROM config WHERE ( "
            + clause + " ) )"
            "AND solver == '" +  id + "' ;";
        // clang-format on
        auto stmt = PrepareAndBind(query, values);
        auto rc   = SQLRety([&]() { return sqlite3_step(stmt.get()); });
        if(rc == SQLITE_DONE)
            return true;
        else
        {
            std::string msg = "Unable to remove database entry: ";
            MIOPEN_LOG_E(msg + SQLErrorMessage());
            return false;
        }
    }

    /// Updates record under key PROBLEM_CONFIG with data ID:VALUES in database.
    /// Returns updated record or boost::none if insertion failed
    template <class T, class V>
    inline boost::optional<DbRecord>
    UpdateUnsafe(const T& problem_config, const std::string& id, const V& values)
    {
        if(dbInvalid)
            return boost::none;
        // UPSERT the value
        {
            std::string clause;
            std::vector<std::string> vals;
            std::tie(clause, vals) = problem_config.InsertQuery();
            auto stmt = PrepareAndBind(clause, vals);
            auto rc   = SQLRety([&]() { return sqlite3_step(stmt.get()); });
            if(rc != SQLITE_DONE)
                MIOPEN_THROW(miopenStatusInternalError,
                             "Failed to insert config: " + SQLErrorMessage());
            auto cnt = sqlite3_changes(ptrDb.get());
            MIOPEN_LOG_I2(cnt << " rows updated");
        }

        // UPSERT perf values
        {
            std::ostringstream params;
            values.Serialize(params);
            std::string clause;
            std::vector<std::string> vals;
            std::tie(clause, vals) = problem_config.WhereClause();

            // clang-format off
            std::string query =
                "INSERT OR REPLACE INTO "
                "perf_db(config, solver, params, arch, num_cu) "
                "VALUES("
                "(SELECT id FROM " + problem_config.table_name() +  " "
                "WHERE ( " + clause + " ) ) , ? , ? , ? , ?);";
            // clang-format on
            vals.push_back(id);
            vals.push_back(params.str());
            vals.push_back(arch);
            vals.push_back(std::to_string(num_cu));
            auto stmt = PrepareAndBind(query, vals);
            auto rc   = SQLRety([&]() { return sqlite3_step(stmt.get()); });
            if(rc != SQLITE_DONE)
            {
                MIOPEN_LOG_E("Failed to insert performance record in the database: " +
                             SQLErrorMessage());
                return boost::none;
            }
        }
        DbRecord record;
        record.SetValues(id, values);
        return record;
    }

    template <class T, class V>
    inline bool StoreRecordUnsafe(const T& problem_config, const std::string& id, const V& values)
    {
        if(dbInvalid)
            return false;
        return bool(UpdateUnsafe(problem_config, id, values));
    }

    /**
     * clears both the config and the associated solver values from the database
     */
    template <class T>
    inline bool ClearRecordUnsafe(const T& problem_config)
    {
        if(dbInvalid)
            return true;
        std::string clause;
        std::vector<std::string> values;
        std::tie(clause, values) = problem_config.WhereClause();
        // clang-format off
        auto query = 
            "DELETE FROM perf_db "
            "WHERE config IN ("
            "SELECT id FROM config WHERE ( "
            + clause + " ))";
        // clang-format on
        auto stmt = PrepareAndBind(query, values);
        auto rc   = SQLRety([&]() { return sqlite3_step(stmt.get()); });
        if(rc != SQLITE_DONE)
        {
            MIOPEN_LOG_E("Unable to Clear databaes entry: " + SQLErrorMessage());
            return false;
        }
        else
            return true;
    }

    /// Searches for record with key PROBLEM_CONFIG and gets VALUES under the ID from it.
    /// Class T should have "void Serialize(PDAttr_t&) const" member function available.
    /// Class V shall have "bool Deserialize(const std::string& str)" member function available.
    ///
    /// Returns false if the problem config is not found in the config table or if there are perf
    /// parameters in the perf_db table
    template <class T, class V>
    inline bool LoadUnsafe(const T& problem_config, const std::string& id, V& values)
    {
        if(dbInvalid)
            return false;
        const auto record = FindRecordUnsafe(problem_config);

        if(!record)
            return false;
        return record->GetValues(id, values);
    }
};
} // namespace miopen
