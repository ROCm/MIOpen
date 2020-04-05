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

SQLitePerfDb::SQLitePerfDb(const std::string& filename_,
                           bool is_system,
                           const std::string& arch_,
                           const std::size_t num_cu_)
    : SQLiteBase(filename_, is_system, arch_, num_cu_)
{
    if(!is_system)
    {
        SQLRes_t res;
        ProblemDescription prob_desc{conv::Direction::Forward};
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
            "`arch` TEXT NOT NULL,"
            "`num_cu` INTEGER NOT NULL,"
            "`params` TEXT NOT NULL"
            ");";
        // clang-format on
        {
            const auto lock = shared_lock(lock_file, GetLockTimeout());
            MIOPEN_VALIDATE_LOCK(lock);
            // clang-format off
            const auto check_tables =
                "SELECT name FROM sqlite_master "
                "WHERE "
                  "type = 'table' AND "
                  "(name = 'config' OR name = 'perf_db');";
            // clang-format on
            SQLExec(check_tables, res);
        }
        if(res.empty())
        {
            const auto lock = exclusive_lock(lock_file, GetLockTimeout());
            MIOPEN_VALIDATE_LOCK(lock);
            if(!SQLExec(create_config + create_perfdb_sql))
                MIOPEN_THROW(miopenStatusInternalError);
            MIOPEN_LOG_I2("Database created successfully");
        }
    }
}
} // namespace miopen
