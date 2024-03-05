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
#include "miopen/bz2.hpp"
#include <miopen/kern_db.hpp>
#include <filesystem>

namespace miopen {
KernDb::KernDb(DbKinds db_kind, const fs::path& filename_, bool is_system_)
    : KernDb(db_kind, filename_, is_system_, compress, decompress)
{
}

KernDb::KernDb(DbKinds db_kind,
               const fs::path& filename_,
               bool is_system_,
               std::function<std::vector<char>(std::vector<char>, bool*)> compress_fn_,
               std::function<std::vector<char>(std::vector<char>, unsigned int)> decompress_fn_)
    : SQLiteBase(db_kind, filename_, is_system_),
      compress_fn(compress_fn_),
      decompress_fn(decompress_fn_)
{
    if(!is_system && DisableUserDbFileIO)
        return;

    if(dbInvalid)
    {
        if(filename.empty())
            MIOPEN_LOG_I("database not present");
        else
            MIOPEN_LOG_I(filename << " database invalid");
        return;
    }
    if(!is_system)
    {
        const std::string create_table = KernelConfig::CreateQuery();
        sql.Exec(create_table);
        MIOPEN_LOG_I2("Database created successfully");
    }
    if(!CheckTableColumns(KernelConfig::table_name(), KernelConfig::FieldNames()))
    {
        std::ostringstream ss;
        ss << "Invalid fields in table: " << KernelConfig::table_name() << " disabling access to "
           << filename;
        MIOPEN_LOG_W(ss.str());
        dbInvalid = true;
    }
}

} // namespace miopen
