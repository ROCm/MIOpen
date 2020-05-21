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
    std::vector<std::string> FieldNames() const
    {
        std::vector<std::string> names;
        Derived::Visit(static_cast<const Derived&>(*this),
                       [&](const std::string& value, const std::string& name) {
                           std::ignore = value;
                           names.push_back(name);
                       });
        Derived::Visit(static_cast<const Derived&>(*this),
                       [&](const int value, const std::string name) {
                           std::ignore = value;
                           names.push_back(name);
                       });

        return names;
    }
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

class SQLite
{
    class impl;
    // do we need propagate const
    std::unique_ptr<impl> pImpl;

    public:
    class Statement
    {
        class impl;
        std::unique_ptr<impl> pImpl;

        public:
        Statement(const SQLite& sql, const std::string& query);
        Statement(const SQLite& sql,
                  const std::string& query,
                  const std::vector<std::string>& vals);
        Statement();
        ~Statement();
        Statement(Statement&&) noexcept;
        Statement& operator=(Statement&&) noexcept;
        Statement& operator=(const Statement&) = delete;
        int Step(const SQLite& sql);
        std::string ColumnText(int idx);
        std::string ColumnBlob(int idx);
        int64_t ColumnInt64(int idx);
        int BindText(int idx, const std::string& txt);
        int BindBlob(int idx, const std::string& blob);
        int BindInt64(int idx, int64_t);
    };

    using result_type = std::vector<std::unordered_map<std::string, std::string>>;
    SQLite();
    SQLite(const std::string& filename_, bool is_system);
    ~SQLite();
    SQLite(SQLite&&) noexcept;
    SQLite& operator=(SQLite&&) noexcept;
    SQLite& operator=(const SQLite&) = delete;
    bool Valid() const;
    result_type Exec(const std::string& query) const;
    int Changes() const;
    int Retry(std::function<int()>) const;
    static int Retry(std::function<int()> f, std::string filename);
    std::string ErrorMessage() const;
};

template <typename Derived>
class SQLiteBase
{
    protected:
    using exclusive_lock = boost::unique_lock<LockFile>;
    using shared_lock    = boost::shared_lock<LockFile>;
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

        if(filename.empty())
        {
            dbInvalid = true;
            return;
        }

        if(!is_system && !filename.empty())
        {
            auto file            = boost::filesystem::path(filename_);
            const auto directory = file.remove_filename();
            if(directory.string().empty())
            {
                dbInvalid = true;
                return;
            }

            if(!(boost::filesystem::exists(directory)))
            {
                if(!boost::filesystem::create_directories(directory))
                    MIOPEN_LOG_W("Unable to create a directory: " << directory);
                else
                    boost::filesystem::permissions(directory, boost::filesystem::all_all);
            }
        }
        sql = std::move(SQLite{filename_, is_system});
        if(!sql.Valid())
        {
            dbInvalid = true;
            if(!is_system)
                MIOPEN_THROW(miopenStatusInternalError, "Cannot open database file:" + filename_);
            else
                MIOPEN_LOG_W("Unable to read system database file:" + filename_ +
                             " Performance may degrade");
        }
        else
            dbInvalid = false;
    }

    static Derived&
    GetCached(const std::string& path, bool is_system, const std::string& arch, std::size_t num_cu);
    // TODO: Fix this for the overhead of having fields per record

    inline auto CheckTableColumns(const std::string& tableName,
                                  const std::vector<std::string>& goldenList) const
    {
        const auto sql_cfg_fds = "PRAGMA table_info(" + tableName + ");";
        SQLite::result_type cfg_res;
        {
            const auto lock = shared_lock(lock_file, GetLockTimeout());
            MIOPEN_VALIDATE_LOCK(lock);
            cfg_res = sql.Exec(sql_cfg_fds);
        }
        std::vector<std::string> cfg_fds(cfg_res.size());
        std::transform(
            cfg_res.begin(), cfg_res.end(), cfg_fds.begin(), [](auto row) { return row["name"]; });
        // search in the golden vector
        bool AllFound = true;
        for(auto& goldenName : goldenList)
        {
            if(std::find(cfg_fds.begin(), cfg_fds.end(), goldenName) == cfg_fds.end())
            {
                AllFound = false;
                std::ostringstream ss;
                ss << "Field " << goldenName << " not found in table: " << tableName;
                MIOPEN_LOG_I2(ss.str());
                // break; Not breaking to enable logging of all missing fields.
            }
        }
        return AllFound;
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

    std::string filename;
    std::string arch;
    size_t num_cu;
    LockFile& lock_file;
    bool dbInvalid;
    SQLite sql;
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
    static constexpr char const* MIOPEN_PERFDB_SCHEMA_VER = "1.0.0";
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
        auto stmt = SQLite::Statement{sql, clause, vals};
        auto rc   = stmt.Step(sql);
        if(rc != SQLITE_DONE)
            MIOPEN_THROW(miopenStatusInternalError,
                         "Failed to insert config: " + sql.ErrorMessage());
        auto cnt = sql.Changes();
        MIOPEN_LOG_I2(cnt << " rows updated");
    }
    template <class T>
    inline std::string GetConfigIDs(const T& prob_desc)
    {
        std::string clause;
        std::vector<std::string> vals;
        std::tie(clause, vals) = prob_desc.WhereClause();
        auto query = "SELECT id FROM " + prob_desc.table_name() + " WHERE ( " + clause + " );";
        auto stmt  = SQLite::Statement{sql, query, vals};
        while(true)
        {
            auto rc = stmt.Step(sql);
            if(rc == SQLITE_ROW)
                return stmt.ColumnText(0);
            else if(rc == SQLITE_DONE)
                return "";
            else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
                MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
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
        auto stmt = SQLite::Statement{sql, select_query, values};
        DbRecord rec;
        while(true)
        {
            auto rc = stmt.Step(sql);
            if(rc == SQLITE_ROW)
                rec.SetValues(stmt.ColumnText(0), stmt.ColumnText(1));
            else if(rc == SQLITE_DONE)
                break;
            else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
                MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
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
        auto stmt = SQLite::Statement{sql, query, values};
        auto rc   = stmt.Step(sql);
        if(rc == SQLITE_DONE)
            return true;
        else
        {
            std::string msg = "Unable to remove database entry: ";
            MIOPEN_LOG_E(msg + sql.ErrorMessage());
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
            auto stmt = SQLite::Statement{sql, clause, vals};
            auto rc   = stmt.Step(sql);
            if(rc != SQLITE_DONE)
                MIOPEN_THROW(miopenStatusInternalError,
                             "Failed to insert config: " + sql.ErrorMessage());
            auto cnt = sql.Changes();
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
            auto stmt = SQLite::Statement{sql, query, vals};
            auto rc   = stmt.Step(sql);
            if(rc != SQLITE_DONE)
            {
                MIOPEN_LOG_E("Failed to insert performance record in the database: " +
                             sql.ErrorMessage());
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
        auto stmt = SQLite::Statement{sql, query, values};
        auto rc   = stmt.Step(sql);
        if(rc != SQLITE_DONE)
        {
            MIOPEN_LOG_E("Unable to Clear databaes entry: " + sql.ErrorMessage());
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
