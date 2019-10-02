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

bool SQLite_Db::IsDBInitialized(const std::string& path)
{
    static std::mutex mutex;
    static const std::lock_guard<std::mutex> lock{mutex};

    static auto init_map = std::unordered_map<std::string, bool>{};

    const auto it = init_map.find(path);
    if(it != init_map.end())
        return true;
    else
    {
        init_map.emplace(path, true);
        return false;
    }
}

SQLite_Db::SQLite_Db(const std::string& filename_,
                     bool is_system,
                     const std::string& arch_,
                     const std::size_t num_cu_)
    : filename(filename_),
      arch(arch_),
      num_cu(std::to_string(num_cu_)),
      lock_file(LockFile::Get(LockFilePath(filename_).c_str()))
{
    const auto db_type = is_system ? "system" : "user";
    MIOPEN_LOG_I2("Initializing " << db_type << " database file " << filename);
    if(!is_system)
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
    int rc = sqlite3_open_v2(
        filename_.c_str(), &ptr_tmp, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if(rc != 0)
    {
        sqlite3_close(ptrDb.get());
        MIOPEN_THROW(miopenStatusInternalError, "Cannot open database file:" + filename_);
    }
    ptrDb = sqlite3_ptr{ptr_tmp};
    if(!is_system)
    {
        const auto lock = exclusive_lock(lock_file, GetLockTimeout());
        MIOPEN_VALIDATE_LOCK(lock);
        if(!IsDBInitialized(filename))
        {
            ProblemDescription prob_desc;
            prob_desc.direction.Set(1);
            prob_desc.in_data_type          = miopenFloat;
            prob_desc.out_data_type         = miopenFloat;
            prob_desc.weights_data_type     = miopenFloat;
            const std::string create_config = prob_desc.CreateQuery();
            // clang-format off
            const std::string create_perfdb_sql =
                "CREATE TABLE  IF NOT EXISTS `perf_db` ("
                        "`id` INTEGER PRIMARY KEY ASC,"
                        "`solver` TEXT NOT NULL,"
                        "`config` INTEGER NOT NULL,"
     //                   "`direction` text NOT NULL,"
                        "`arch` TEXT NOT NULL,"
                        "`num_cu` INTEGER NOT NULL,"
                        "`params` TEXT NOT NULL"
                        ");";
            // clang-format on
            if(!SQLExec(create_config + create_perfdb_sql))
                MIOPEN_THROW(miopenStatusInternalError);
        }
        MIOPEN_LOG_I2("Database created successfully");
    }
}
} // namespace miopen
