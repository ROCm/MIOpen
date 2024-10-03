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

#include <miopen/config.hpp>

#if !MIOPEN_ENABLE_SQLITE
#error "MIOPEN_ENABLE_SQLITE = Off"
#endif

#include <miopen/db_record.hpp>
#include <miopen/db.hpp>
#include <miopen/manage_ptr.hpp>
#include <miopen/errors.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/env.hpp>

#include <boost/core/explicit_operator_bool.hpp>
#include <boost/none.hpp>
#include <boost/optional/optional.hpp>
#include "sqlite3.h"
#include <mutex>
#include <thread>

#include <string>
#include <chrono>
#include <unordered_map>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_DISABLE_SQL_WAL)
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEBUG_PERFDB_OVERRIDE)

namespace miopen {

constexpr bool InMemDb = MIOPEN_EMBED_DB;
#if MIOPEN_ENABLE_SQLITE_BACKOFF
const auto MIOPEN_SQL_BUSY_TIMEOUT_MS = 10;
#else
const auto MIOPEN_SQL_BUSY_TIMEOUT_MS = 60000;
#endif
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
                       [&](const int64_t value, const std::string name) {
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
                       [&](const int64_t value, const std::string name) {
                           clauses.push_back("(" + name + " = ? )");
                           values.push_back(std::to_string(value));
                       });
        const std::string clause = JoinStrings(clauses, " AND ");
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
                       [&](const int64_t value, const std::string name) {
                           int_names.push_back(name);
                           values.push_back(std::to_string(value));
                       });
        const std::vector<std::string> tokens((values.size()), "?");
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
                       [&](const int64_t value, const std::string name) {
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

class MIOPEN_INTERNALS_EXPORT SQLite
{
    class impl;
    // do we need propagate const
    std::unique_ptr<impl> pImpl;

public:
    class MIOPEN_INTERNALS_EXPORT Statement
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
        std::vector<char> ColumnBlob(int idx);
        int64_t ColumnInt64(int idx);
        int BindText(int idx, const std::string& txt);
        int BindPath(int idx, const fs::path& path);
        int BindBlob(int idx, const std::vector<char>& blob);
        int BindInt64(int idx, int64_t);
    };

    using result_type = std::vector<std::unordered_map<std::string, std::string>>;
    SQLite();
    SQLite(const fs::path& filename_, bool is_system);
    ~SQLite();
    SQLite(SQLite&&) noexcept;
    SQLite& operator=(SQLite&&) noexcept;
    SQLite& operator=(const SQLite&) = delete;
    bool Valid() const;
    result_type Exec(const std::string& query) const;
    int Changes() const;
    int Retry(std::function<int()>) const;
    static int Retry(std::function<int()> f, fs::path filename);
    std::string ErrorMessage() const;
};

template <typename Derived>
class SQLiteBase
{
protected:
public:
    SQLiteBase(DbKinds, const fs::path& filename_, bool is_system_)
        : filename(filename_), is_system(is_system_)
    {
        if(DisableUserDbFileIO && !is_system)
            return;

        MIOPEN_LOG_I2("Initializing " << (InMemDb ? "In Memory " : "")
                                      << (is_system ? "system" : "user") << " database file "
                                      << filename);

        if(filename.empty())
        {
            dbInvalid = true;
            return;
        }
        else if(!is_system)
        {
            if(!filename_.has_parent_path())
            {
                dbInvalid = true;
                return;
            }

            auto directory = filename_.parent_path();
            if(!fs::exists(directory))
            {
                if(!fs::create_directories(directory))
                    MIOPEN_LOG_W("Unable to create a directory: " << directory);
                else
                    fs::permissions(directory, FS_ENUM_PERMS_ALL);
            }
        }
        sql = SQLite{filename_, is_system};
        if(!sql.Valid())
        {
            bool isKDB = filename.extension() == ".kdb";
            dbInvalid  = true;
            filename   = "";
            if(!is_system)
            {
                MIOPEN_THROW(miopenStatusInternalError, "Cannot open database file:" + filename_);
            }
            else
            {
                const auto log_level =
                    (!MIOPEN_DISABLE_SYSDB) ? LoggingLevel::Warning : LoggingLevel::Info;
                if(isKDB && (log_level == LoggingLevel::Warning))
                {
                    static const auto kdb_message_issued = [&]() {
                        MIOPEN_LOG_W(
                            "Missing system database file: "
                            << filename_
                            << " Performance may degrade. Please follow instructions to install: "
                               "https://github.com/ROCm/"
                               "MIOpen#installing-miopen-kernels-package");
                        return true;
                    }();
                    std::ignore = kdb_message_issued;
                }
                else
                {
                    MIOPEN_LOG(log_level,
                               "Unable to read system database file:"
                                   << filename_ << " Performance may degrade");
                }
            }
        }
        else
        {
            dbInvalid = false;
            if(!is_system && !env::enabled(MIOPEN_DEBUG_DISABLE_SQL_WAL))
            {
                auto res = sql.Exec("PRAGMA journal_mode=WAL;");
                if(res.empty() || res[0]["journal_mode"] != "wal")
                {
                    MIOPEN_LOG_I("SQLite does not support WAL");
                }
            }
        }
    }

    static Derived& GetCached(const fs::path& path, bool is_system);
    // TODO: Fix this for the overhead of having fields per record

    inline auto CheckTableColumns(const std::string& tableName,
                                  const std::vector<std::string>& goldenList) const
    {
        const auto sql_cfg_fds = "PRAGMA table_info(" + tableName + ");";
        SQLite::result_type cfg_res;
        cfg_res = sql.Exec(sql_cfg_fds);
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
                MIOPEN_LOG_I2("Field " << goldenName << " not found in table: " << tableName);
                // break; Not breaking to enable logging of all missing fields.
            }
        }
        return AllFound;
    }
    template <typename... U>
    inline auto FindRecord(U&... args)
    {
        using Ret = decltype(reinterpret_cast<Derived*>(this)->FindRecordUnsafe(args...));
        if(!is_system && DisableUserDbFileIO)
            return Ret{};
        return reinterpret_cast<Derived*>(this)->FindRecordUnsafe(args...);
    }

    template <typename... U>
    inline auto RemoveRecord(U&... args)
    {
        if(!is_system && DisableUserDbFileIO)
            return true;
        return reinterpret_cast<Derived*>(this)->RemoveRecordUnsafe(args...);
    }

    template <typename... U>
    inline auto StoreRecord(U&... args)
    {
        if(!is_system && DisableUserDbFileIO)
            return true;
        return reinterpret_cast<Derived*>(this)->StoreRecordUnsafe(args...);
    }

    template <typename... U>
    inline auto Remove(const U&... args)
    {
        if(!is_system && DisableUserDbFileIO)
            return true;
        return reinterpret_cast<Derived*>(this)->RemoveUnsafe(args...);
    }

    template <typename... U>
    inline auto Update(const U&... args)
    {
        using Ret = decltype(reinterpret_cast<Derived*>(this)->UpdateUnsafe(args...));
        if(!is_system && DisableUserDbFileIO)
            return Ret{};
        return reinterpret_cast<Derived*>(this)->UpdateUnsafe(args...);
    }

    template <typename... U>
    inline bool Load(U&&... args)
    {
        if(!is_system && DisableUserDbFileIO)
            return false;
        return reinterpret_cast<Derived*>(this)->LoadUnsafe(args...);
    }

    fs::path filename;
    bool dbInvalid;
    SQLite sql;
    bool is_system;
};

template <typename Derived>
Derived& SQLiteBase<Derived>::GetCached(const fs::path& path, bool is_system)
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock{mutex};

    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static auto instances = std::map<fs::path, Derived>{};
    const auto it         = instances.find(path);

    if(it != instances.end())
        return it->second;

    instances.emplace(path, Derived{path, is_system});
    return instances.at(path);
}

class SQLitePerfDb : public SQLiteBase<SQLitePerfDb>
{
public:
    static constexpr char const* MIOPEN_PERFDB_SCHEMA_VER = "1.1.0";
    MIOPEN_INTERNALS_EXPORT
    SQLitePerfDb(DbKinds db_kind, const fs::path& filename_, bool is_system);

    template <class T>
    inline void InsertConfig(const T& prob_desc)
    {
        std::string clause;
        std::vector<std::string> vals;
        std::tie(clause, vals) = prob_desc.InsertQuery();
        auto stmt              = SQLite::Statement{sql, clause, vals};
        auto rc                = stmt.Step(sql);
        if(rc != SQLITE_DONE)
        {
            MIOPEN_THROW(miopenStatusInternalError,
                         "Failed to insert config: " + sql.ErrorMessage());
        }
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
            {
                return stmt.ColumnText(0);
            }
            else if(rc == SQLITE_DONE)
            {
                return "";
            }
            else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
            {
                MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
            }
        }
    }
    template <typename T>
    inline boost::optional<DbRecord> FindRecordUnsafe(const T& problem_config)
    {
        if(dbInvalid)
            return boost::none;

        const auto& pdb_ovr = env::value(MIOPEN_DEBUG_PERFDB_OVERRIDE);
        if(!pdb_ovr.empty())
        {
            MIOPEN_LOG_I2("overriding tuning params with: " << pdb_ovr);
            DbRecord ovr_rec;
            const auto solv_vals = SplitDelim(pdb_ovr, ':');
            bool success         = true;
            for(const auto& solv_val : solv_vals)
            {
                const auto vals = SplitDelim(solv_val, ';');
                if(vals.size() != 2)
                {
                    MIOPEN_LOG_W("Invalid value for MIOPEN_DEBUG_PERFDB_OVERRIDE. Format: "
                                 "<solver1_name>;<params>:<solver2_name>;params");
                    success = false;
                    break;
                }
                MIOPEN_LOG_I2("Inserting Overriding PDB entry: " << vals[0] << ";" << vals[1]);
                ovr_rec.SetValues(vals.at(0), vals.at(1));
            }
            if(success)
                return {ovr_rec};
        }
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
            "( " + clause + " );";
        // clang-format on
        auto stmt = SQLite::Statement{sql, select_query, values};
        DbRecord rec;
        while(true)
        {
            auto rc = stmt.Step(sql);
            if(rc == SQLITE_ROW)
            {
                rec.SetValues(stmt.ColumnText(0), stmt.ColumnText(1));
            }
            else if(rc == SQLITE_DONE)
            {
                break;
            }
            else if(rc == SQLITE_ERROR || rc == SQLITE_MISUSE)
            {
                MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
            }
        }
        if(rec.GetSize() == 0)
            return boost::none;
        else
            return {rec};
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
        {
            return true;
        }
        else
        {
            const std::string msg = "Unable to remove database entry: ";
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
            auto stmt              = SQLite::Statement{sql, clause, vals};
            auto rc                = stmt.Step(sql);
            if(rc != SQLITE_DONE)
            {
                MIOPEN_THROW(miopenStatusInternalError,
                             "Failed to insert config: " + sql.ErrorMessage());
            }
            auto cnt = sql.Changes();
            MIOPEN_LOG_I2(cnt << " rows updated");
        }

        // UPSERT perf values
        {
            std::ostringstream params;
            values.Serialize(params);
            std::string clause;
            std::vector<std::string> vals(2);
            std::tie(clause, vals) = problem_config.WhereClause();

            // clang-format off
            std::string query =
                "INSERT OR REPLACE INTO "
                "perf_db(config, solver, params) "
                "VALUES("
                "(SELECT id FROM " + problem_config.table_name() +  " "
                "WHERE ( " + clause + " ) ) , ? , ?);";
            // clang-format on
            vals.push_back(id);
            vals.push_back(params.str());
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
