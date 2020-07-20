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
#include <miopen/sqlite_db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/errors.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/logger.hpp>
#include <miopen/md5.hpp>
#include <miopen/problem_description.hpp>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/none.hpp>
#include <boost/optional.hpp>

#include <memory>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <ios>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace miopen {

class SQLite::impl
{
    struct SQLiteCloser
    {
        void operator()(sqlite3* ptr)
        {
            const auto c_filename = sqlite3_db_filename(ptr, "main");
            std::string filename_((c_filename == nullptr) ? "" : c_filename);
            SQLite::Retry([&]() { return sqlite3_close(ptr); }, filename_);
        }
    };

    public:
    impl(const std::string& filename_, bool is_system)
    {
        sqlite3* ptr_tmp;
        int rc = 0;
        if(is_system)
            rc = sqlite3_open_v2(filename_.c_str(), &ptr_tmp, SQLITE_OPEN_READONLY, nullptr);
        else
            rc = sqlite3_open_v2(
                filename_.c_str(), &ptr_tmp, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
        ptrDb = sqlite3_ptr{ptr_tmp};
        sqlite3_busy_timeout(ptrDb.get(), MIOPEN_SQL_BUSY_TIMEOUT_MS);
        isValid = (rc == 0);
    }

    using sqlite3_ptr = std::unique_ptr<sqlite3, SQLiteCloser>;
    sqlite3_ptr ptrDb = nullptr;
    bool isValid;
};

static int find_callback(void* _res, int argc, char** argv, char** azColName)
{
    SQLite::result_type* res = static_cast<SQLite::result_type*>(_res);
    std::unordered_map<std::string, std::string> record;
    for(auto i               = 0; i < argc; i++)
        record[azColName[i]] = (argv[i] != nullptr) ? argv[i] : "NULL";
    if(res != nullptr)
        res->push_back(record);
    return 0;
}

SQLite::SQLite() : pImpl(nullptr) {}
SQLite::~SQLite()                 = default;
SQLite::SQLite(SQLite&&) noexcept = default;
SQLite& SQLite::operator=(SQLite&&) noexcept = default;
SQLite::result_type SQLite::Exec(const std::string& query) const
{
    SQLite::result_type res;
    MIOPEN_LOG_T(std::this_thread::get_id() << ":" << query);
    {
        auto rc = Retry([&]() {
            return sqlite3_exec(pImpl->ptrDb.get(),
                                query.c_str(),
                                find_callback,
                                static_cast<void*>(&res),
                                nullptr);
        });
        if(rc != SQLITE_OK)
        {
            MIOPEN_LOG_I2(query);
            MIOPEN_THROW(miopenStatusInternalError, ErrorMessage());
        }
    }
    return res;
}

int SQLite::Retry(std::function<int()> f, std::string filename)
{
    int rc = f();
    if(rc == SQLITE_BUSY)
    {
        MIOPEN_THROW("Timeout while waiting for Database: " + filename);
    }
    else
        return rc;
}

int SQLite::Retry(std::function<int()> f) const
{
    std::string filename(sqlite3_db_filename(pImpl->ptrDb.get(), "main"));
    return SQLite::Retry(f, filename);
}

int SQLite::Changes() const { return sqlite3_changes(pImpl->ptrDb.get()); }

std::string SQLite::ErrorMessage() const
{
    std::string errMsg = "Internal error while accessing SQLite database: ";
    return errMsg + sqlite3_errmsg(pImpl->ptrDb.get());
}
bool SQLite::Valid() const { return pImpl->isValid; }

class SQLite::Statement::impl
{
    using sqlite3_stmt_ptr = MIOPEN_MANAGE_PTR(sqlite3_stmt*, sqlite3_finalize);
    sqlite3_stmt_ptr Prepare(const SQLite& sql, const std::string& query)
    {
        sqlite3_stmt* ptr = nullptr;
        MIOPEN_LOG_I2(query);
        auto rc =
            sqlite3_prepare_v2(sql.pImpl->ptrDb.get(), query.c_str(), query.size(), &ptr, nullptr);
        if(rc != SQLITE_OK)
        {
            std::string err_msg = "SQLite prepare error: ";
            MIOPEN_THROW(miopenStatusInternalError, err_msg + sql.ErrorMessage());
        }
        return sqlite3_stmt_ptr{ptr};
    }

    public:
    impl(const SQLite& sql, const std::string& query) { ptrStmt = Prepare(sql, query); }
    impl(const SQLite& sql, const std::string& query, const std::vector<std::string>& vals)
    {
        ptrStmt = Prepare(sql, query);
        int cnt = 1;
        for(auto& kinder : vals)
        {
            auto rc = sqlite3_bind_text(
                ptrStmt.get(), cnt++, kinder.data(), kinder.size(), SQLITE_TRANSIENT); // NOLINT
            if(rc != SQLITE_OK)
                MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
        }
        MIOPEN_LOG_I2("[" << JoinStrings(vals, ",") << "]");
    }

    sqlite3_stmt_ptr ptrStmt = nullptr;
};

SQLite::SQLite(const std::string& filename_, bool is_system)
    : pImpl{std::make_unique<impl>(filename_, is_system)}
{
}

SQLite::Statement::Statement(const SQLite& sql, const std::string& query)
    : pImpl{std::make_unique<impl>(sql, query)}
{
}
SQLite::Statement::Statement(const SQLite& sql,
                             const std::string& query,
                             const std::vector<std::string>& vals)
    : pImpl{std::make_unique<impl>(sql, query, vals)}
{
}
SQLite::Statement::~Statement() = default;
SQLite::Statement::Statement() : pImpl{nullptr} {}
SQLite::Statement::Statement(Statement&&) noexcept = default;
SQLite::Statement& SQLite::Statement::operator=(Statement&&) noexcept = default;
int SQLite::Statement::Step(const SQLite& sql)
{
    return sql.Retry([&]() { return sqlite3_step(pImpl->ptrStmt.get()); });
}
std::string SQLite::Statement::ColumnText(int idx)
{
    size_t bytes = sqlite3_column_bytes(pImpl->ptrStmt.get(), idx);
    return std::string{
        reinterpret_cast<const char*>(sqlite3_column_text(pImpl->ptrStmt.get(), idx)), bytes};
}

std::string SQLite::Statement::ColumnBlob(int idx)
{
    auto ptr = sqlite3_column_blob(pImpl->ptrStmt.get(), idx);
    auto sz  = sqlite3_column_bytes(pImpl->ptrStmt.get(), idx);
    return std::string{reinterpret_cast<const char*>(ptr), static_cast<size_t>(sz)};
}
int64_t SQLite::Statement::ColumnInt64(int idx)
{
    return sqlite3_column_int64(pImpl->ptrStmt.get(), idx);
}

int SQLite::Statement::BindText(int idx, const std::string& txt)
{
    sqlite3_bind_text(
        pImpl->ptrStmt.get(), idx, txt.data(), txt.size(), SQLITE_TRANSIENT); // NOLINT
    return 0;
}
int SQLite::Statement::BindBlob(int idx, const std::string& blob)
{
    sqlite3_bind_blob(
        pImpl->ptrStmt.get(), idx, blob.data(), blob.size(), SQLITE_TRANSIENT); // NOLINT
    return 0;
}

int SQLite::Statement::BindInt64(int idx, const int64_t num)
{
    sqlite3_bind_int64(pImpl->ptrStmt.get(), idx, num);
    return 0;
}

SQLitePerfDb::SQLitePerfDb(const std::string& filename_,
                           bool is_system,
                           const std::string& arch_,
                           const std::size_t num_cu_)
    : SQLiteBase(filename_, is_system, arch_, num_cu_)
{
    if(dbInvalid)
    {
        if(filename.empty())
            MIOPEN_LOG_I("database not present");
        else
            MIOPEN_LOG_I(filename + " database invalid");
        return;
    }
    ProblemDescription prob_desc{conv::Direction::Forward};
    prob_desc.in_data_type      = miopenFloat;
    prob_desc.out_data_type     = miopenFloat;
    prob_desc.weights_data_type = miopenFloat;
    if(!is_system)
    {
        SQLite::result_type res;
        const std::string create_config_sql = prob_desc.CreateQuery();
        // clang-format off
        const std::string create_perfdb_sql =
            "CREATE TABLE  IF NOT EXISTS `perf_db` ("
            "`id` INTEGER PRIMARY KEY ASC,"
            "`solver` TEXT NOT NULL,"
            "`config` INTEGER NOT NULL,"
            "`arch` TEXT NOT NULL,"
            "`num_cu` INTEGER NOT NULL,"
            "`params` TEXT NOT NULL"
            ");"
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "`idx_perf_db` "
            "ON perf_db(solver, config, arch, num_cu);";

        // clang-format on
        {
            // clang-format off
            const auto check_tables =
                "SELECT name FROM sqlite_master "
                "WHERE "
                  "type = 'table' AND "
                  "(name = 'config');";
            // clang-format on
            res = sql.Exec(check_tables);
            if(res.empty())
            {
                sql.Exec(create_config_sql);
            }
        }
        {
            // clang-format off
            const auto check_tables =
                "SELECT name FROM sqlite_master "
                "WHERE "
                  "type = 'table' AND "
                  "(name = 'perf_db');";
            // clang-format on
            res = sql.Exec(check_tables);
            if(res.empty())
            {
                sql.Exec(create_perfdb_sql);
            }
        }
        MIOPEN_LOG_T("Database created successfully");
    }
    // Check fields for the tables
    if(!dbInvalid)
    {
        if(!CheckTableColumns(ProblemDescription::table_name(), prob_desc.FieldNames()))
        {
            std::ostringstream ss;
            ss << "Invalid fields in table: " << ProblemDescription::table_name()
               << " disabling access to " << filename;
            MIOPEN_LOG_W(ss.str());
            dbInvalid = true;
        }
        if(!CheckTableColumns("perf_db", {"solver", "config", "arch", "num_cu", "params"}))
        {
            MIOPEN_LOG_W("Invalid fields in table: perf_db disabling access to " + filename);
            dbInvalid = true;
        }
    }
}
} // namespace miopen
