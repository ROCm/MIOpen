/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_KERN_DB_HPP_
#define GUARD_MIOPEN_KERN_DB_HPP_

#include <miopen/sqlite_db.hpp>

#include <boost/core/explicit_operator_bool.hpp>
#include <boost/none.hpp>
#include <boost/optional/optional.hpp>

#include <string>
#include <chrono>
#include <thread>

namespace boost {
namespace filesystem {
class path;
} // namespace filesystem
} // namespace boost

namespace miopen {

struct KernelConfig
{
    static std::string table_name() { return "kern_db"; }
    std::string kernel_name;
    std::string kernel_args;
    std::string kernel_blob;
    static std::string CreateQuery()
    {
        std::ostringstream ss;
        ss << "CREATE TABLE IF NOT EXISTS `" << KernelConfig::table_name() << "` ("
           << "`id` INTEGER PRIMARY KEY ASC"
           << ",`kernel_name` TEXT NOT NULL"
           << ",`kernel_args` TEXT NOT NULL"
           << ",`kernel_blob` BLOB NOT NULL"
           << ");"
           << "CREATE UNIQUE INDEX IF NOT EXISTS "
           << "`idx_" << KernelConfig::table_name() << "` "
           << "ON " << KernelConfig::table_name() << "(kernel_name, kernel_args);";
        return ss.str();
    }
    std::string Where() const
    {
        std::ostringstream ss;
        ss << "(kernel_name = '" << kernel_name << "')"
           << " AND (kernel_args = '" << kernel_args << "')";
        return ss.str();
    }
};

class KernDb : public SQLiteBase<KernDb>
{
    public:
    KernDb(const std::string& filename_,
           bool is_system,
           const std::string& arch,
           std::size_t num_cu);
    template <typename T>
    bool RemoveRecordUnsafe(const T& problem_config)
    {
        if(filename.empty())
            return true;
        auto del_query =
            "DELETE FROM " + T::table_name() + " WHERE " + problem_config.Where() + ";";
        sqlite3_stmt_ptr pStmt = Prepare(del_query);
        auto rc                = SQLRety([&]() { return sqlite3_step(pStmt.get()); });
        if(rc == SQLITE_DONE)
            return true;
        else
        {
            MIOPEN_THROW(miopenStatusInternalError, SQLErrorMessage());
            return false;
        }
    }

    template <typename T>
    boost::optional<std::string> FindRecordUnsafe(const T& problem_config)
    {
        if(filename.empty())
            return boost::none;
        // Where clause with inserted values defeats the purpose of a prepraed statement
        auto select_query =
            "SELECT kernel_blob FROM " + T::table_name() + " WHERE " + problem_config.Where() + ";";
        sqlite3_stmt_ptr pStmt = Prepare(select_query);
        // only one result field
        // assert one row
        auto rc = SQLRety([&]() { return sqlite3_step(pStmt.get()); });
        if(rc == SQLITE_ROW)
        {
            auto ptr = sqlite3_column_blob(pStmt.get(), 0);
            auto sz  = sqlite3_column_bytes(pStmt.get(), 0);
            std::string blob(reinterpret_cast<const char*>(ptr), sz);
            return blob;
        }
        else if(rc == SQLITE_DONE)
            return boost::none;
        else
            MIOPEN_THROW(miopenStatusInternalError, SQLErrorMessage());
        return boost::none;
    }

    template <typename T>
    boost::optional<std::string> StoreRecordUnsafe(const T& problem_config)
    {
        if(filename.empty())
            return boost::none;
        auto insert_query = "INSERT OR REPLACE INTO " + T::table_name() +
                            "(kernel_name, kernel_args, kernel_blob) VALUES(?, ?, ?);";
        sqlite3_stmt_ptr pStmt = Prepare(insert_query);
        sqlite3_bind_text(pStmt.get(),
                          1,
                          problem_config.kernel_name.data(),
                          problem_config.kernel_name.size(),
                          SQLITE_TRANSIENT); // NOLINT
        sqlite3_bind_text(pStmt.get(),
                          2,
                          problem_config.kernel_args.data(),
                          problem_config.kernel_args.size(),
                          SQLITE_TRANSIENT); // NOLINT
        sqlite3_bind_blob(pStmt.get(),
                          3,
                          problem_config.kernel_blob.data(),
                          problem_config.kernel_blob.size(),
                          SQLITE_TRANSIENT); // NOLINT
        auto rc = SQLRety([&]() { return sqlite3_step(pStmt.get()); });
        if(rc != SQLITE_DONE)
            MIOPEN_THROW(miopenStatusInternalError, SQLErrorMessage());
        return problem_config.kernel_blob;
    }
};
} // namespace miopen

#endif // GUARD_MIOPEN_KERN_DB_HPP_
