/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include "test.hpp"
#include "driver.hpp"
#include <miopen/sqlite_db.hpp>
#include <miopen/temp_file.hpp>

const char* const lfs_db = R"(version https://git-lfs.github.com/spec/v1
oid sha256:cc45c32e44560074b5e4b0c0e48472a86e6b3bb1c73c189580f950f098d2a8d7
size 357490688)";

struct DummyDB
{
};

bool test_lfs_db(bool is_system)
{
    miopen::TempFile tmp_db{"test_lfs_db"};
    // write file to temp file
    std::ofstream tmp_db_file(tmp_db.Path());
    tmp_db_file << lfs_db;
    tmp_db_file.close();
    // construct a db out of it
    miopen::SQLiteBase<DummyDB> lfs_sqdb{miopen::DbKinds::PerfDb, tmp_db, is_system};
    return lfs_sqdb.dbInvalid;
}

int main(int argc, char* argv[])
{
    std::ignore = argc;
    std::ignore = argv;

    CHECK(test_lfs_db(
        true)); // System DB should pass, since the lfs file was installed in the sys directory
// Embedded user dbs ignore the filename and just creates an in-memory database
#if !MIOPEN_EMBED_DB
    CHECK(throws(std::bind(
        test_lfs_db, false))); // User db should fail since MIOpen should not create such a file
                               // ever, if it exists its a corrupt file which should be reported.
#endif
}
