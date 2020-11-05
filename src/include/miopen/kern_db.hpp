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

#include <miopen/config.h>

#if MIOPEN_ENABLE_SQLITE

#include <miopen/sqlite_db.hpp>
#include <miopen/bz2.hpp>
#include <miopen/md5.hpp>
#include <miopen/exp_backoff.hpp>

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
    static std::vector<std::string> FieldNames()
    {
        return {"kernel_name", "kernel_args", "kernel_blob"};
    }
    static std::string CreateQuery()
    {
        std::ostringstream ss;
        ss << "CREATE TABLE IF NOT EXISTS `" << KernelConfig::table_name() << "` ("
           << "`id` INTEGER PRIMARY KEY ASC"
           << ",`kernel_name` TEXT NOT NULL"
           << ",`kernel_args` TEXT NOT NULL"
           << ",`kernel_blob` BLOB NOT NULL"
           << ",`kernel_hash` TEXT NOT NULL"
           << ",`uncompressed_size` INT NOT NULL"
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
    std::function<std::string(std::string, bool*)> compress_fn;
    std::function<std::string(std::string, unsigned int)> decompress_fn;

    public:
    KernDb(const std::string& filename_,
           bool is_system,
           const std::string& arch,
           std::size_t num_cu);
    // This constructor is only intended for testing
    KernDb(const std::string& filename_,
           bool _is_system,
           const std::string& _arch,
           std::size_t _num_cu,
           std::function<std::string(std::string, bool*)> _compress_fn,
           std::function<std::string(std::string, unsigned int)> _decompress_fn);
    template <typename T>
    bool RemoveRecordUnsafe(const T& problem_config)
    {
        if(filename.empty())
            return true;
        auto del_query =
            "DELETE FROM " + T::table_name() + " WHERE " + problem_config.Where() + ";";
        auto stmt = SQLite::Statement{sql, del_query};
        auto rc   = stmt.Step(sql);
        if(rc == SQLITE_DONE)
            return true;
        else
        {
            MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
            return false;
        }
    }

    template <typename T>
    bool MarkRecordUnsafe(const T& problem_config)
    {
        if(filename.empty())
            return true;
        auto upd_query = "UPDATE " + T::table_name() + " SET uncompressed_size = -1 WHERE " +
                         problem_config.Where() + ";";
        auto stmt = SQLite::Statement{sql, upd_query};
        auto rc   = stmt.Step(sql);
        if(rc == SQLITE_DONE)
            return true;
        else
            MIOPEN_THROW(miopenStatusInternalError, "Unable to update binary cache");

        return false;
    }

    template <typename T>
    boost::optional<std::string> FindRecordUnsafe(const T& problem_config)
    {
        if(filename.empty())
            return boost::none;
        // Where clause with inserted values defeats the purpose of a prepraed statement
        auto select_query = "SELECT kernel_blob, kernel_hash, uncompressed_size FROM " +
                            T::table_name() + " WHERE " + problem_config.Where() + ";";

        LazyExponentialBackoff exp_bo{10, 2, std::chrono::seconds(60)};
        while(exp_bo)
        {
            auto stmt = SQLite::Statement{sql, select_query};
            // only one result field
            // assert one row
            auto rc = stmt.Step(sql);
            if(rc == SQLITE_ROW)
            {
                auto uncompressed_size = stmt.ColumnInt64(2);
                if(uncompressed_size == -1)
                {
                    auto slot = *exp_bo;
                    // sleep for sometime and then try again
                    if(slot != 0)
                        std::this_thread::sleep_for(std::chrono::microseconds(100 * slot));

                    continue;
                }

                auto compressed_blob = stmt.ColumnBlob(0);
                auto md5_hash        = stmt.ColumnText(1);

                std::string& decompressed_blob = compressed_blob;
                if(uncompressed_size != 0)
                {
                    decompressed_blob = decompress_fn(compressed_blob, uncompressed_size);
                }
                auto new_md5 = md5(decompressed_blob);
                if(new_md5 != md5_hash)
                    MIOPEN_THROW(miopenStatusInternalError, "Possible database corruption");
                return decompressed_blob;
            }
            else if(rc == SQLITE_DONE)
                return boost::none;
            else
                MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
        }
        MIOPEN_THROW(miopenStatusInternalError, "Timeout waiting for code object to compile");
        return boost::none;
    }

    template <typename T>
    boost::optional<std::string> StoreRecordUnsafe(const T& problem_config)
    {
        if(filename.empty())
            return boost::none;
        auto insert_query = "INSERT OR REPLACE INTO " + T::table_name() +
                            "(kernel_name, kernel_args, kernel_blob, kernel_hash, "
                            "uncompressed_size) VALUES(?, ?, ?, ?, ?);";
        auto md5_sum           = md5(problem_config.kernel_blob);
        auto uncompressed_size = problem_config.kernel_blob.size();
        bool success           = false;
        auto compressed_blob   = compress_fn(problem_config.kernel_blob, &success);
        auto stmt              = SQLite::Statement{sql, insert_query};
        stmt.BindText(1, problem_config.kernel_name);
        stmt.BindText(2, problem_config.kernel_args);
        if(!success)
        {
            stmt.BindBlob(3, problem_config.kernel_blob);
            stmt.BindInt64(5, 0);
        }
        else
        {
            stmt.BindBlob(3, compressed_blob);
            stmt.BindInt64(5, uncompressed_size);
        }
        stmt.BindText(4, md5_sum);

        auto rc = stmt.Step(sql);
        if(rc != SQLITE_DONE)
            MIOPEN_THROW(miopenStatusInternalError, sql.ErrorMessage());
        return problem_config.kernel_blob;
    }
};
} // namespace miopen
#endif
#endif // GUARD_MIOPEN_KERN_DB_HPP_
