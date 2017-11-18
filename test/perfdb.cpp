/*******************************************************************************
*
* MIT License
*
* Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "miopen/db_record.hpp"
#include "temp_file_path.hpp"
#include "test.hpp"

namespace miopen {
namespace tests {

struct TestData
{
    int x;
    int y;

    TestData()
    {
        const int seed = 42;

        static std::mt19937 rng(seed);
        static std::uniform_int_distribution<std::mt19937::result_type> dist{};

        x = dist(rng);
        y = dist(rng);
    }

    TestData(int x_, int y_) : x(x_), y(y_) {}

    void Serialize(std::ostream& s) const
    {
        static const auto sep = ',';
        s << x << sep << y;
    }

    bool Deserialize(const std::string& s)
    {
        static const auto sep = ',';
        TestData t;
        std::istringstream ss(s);

        auto success = DeserializeField(ss, &t.x, sep) && DeserializeField(ss, &t.y, sep);

        if(!success)
            return false;

        *this = t;
        return true;
    }

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    void LegacySerialize(std::ostream& s) const
    {
        Serialize(s);
        s << ",l";
    }

    bool LegacyDeserialize(const std::string& s) { return Deserialize(s); }
#endif
    bool operator==(const TestData& other) const { return x == other.x && y == other.y; }

    private:
    inline static bool DeserializeField(std::istream& from, int* ret, char separator)
    {
        std::string part;

        if(!std::getline(from, part, separator))
            return false;

        const auto start = part.c_str();
        char* end;
        auto value = std::strtol(start, &end, 10);

        if(start == end)
            return false;

        *ret = value;
        return true;
    }
};

std::ostream& operator<<(std::ostream& s, const TestData& td)
{
    s << "x: " << td.x << ", y: " << td.y;
    return s;
}

class DbRecordTest
{
    public:
    DbRecordTest() : _temp_file_path("/tmp/miopen.tests.perfdb.XXXXXX") {}
    virtual ~DbRecordTest() {}

    protected:
    static const TestData& key()
    {
        static const TestData data(1, 2);
        return data;
    }

    static const TestData& value0()
    {
        static const TestData data(3, 4);
        return data;
    }

    static const TestData& value1()
    {
        static const TestData data(5, 6);
        return data;
    }

    static const char* id0() { return "0"; }
    static const char* id1() { return "1"; }
    static const char* missing_id() { return "2"; }
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    static const char* legacy_id() { return "ConvOclDirectFwd"; } // const from db_record.cpp
#endif

    const char* temp_file_path() const { return _temp_file_path; }

    private:
    TempFilePath _temp_file_path;
};

class DbRecordReadTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        std::ostringstream ss_vals;
        ss_vals << key().x << ',' << key().y << '=' << id1() << ':' << value1().x << ','
                << value1().y << ';' << id0() << ':' << value0().x << ',' << value0().y;

        std::ofstream(temp_file_path()) << ss_vals.str() << std::endl;

        TestData read0, read1;

        {
            DbRecord record(temp_file_path(), key());

            EXPECT(record.Load(id0(), read0));
            EXPECT(record.Load(id1(), read1));
        }

        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
    }
};

class DbRecordWriteTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        std::ostringstream ss_vals;
        ss_vals << key().x << ',' << key().y << '=' << id1() << ':' << value1().x << ','
                << value1().y << ';' << id0() << ':' << value0().x << ',' << value0().y;

        (void)std::ofstream(temp_file_path());

        {
            DbRecord record(temp_file_path(), key());

            EXPECT(record.Store(id0(), value0()));
            EXPECT(record.Store(id1(), value1()));
        }

        std::string read;

        EXPECT(std::getline(std::ifstream(temp_file_path()), read).good());
        EXPECT_EQUAL(read, ss_vals.str());
    }
};

#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
class DbRecordLegacyReadTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        std::ostringstream ss_vals;
        ss_vals << key().x << ',' << key().y << ",l " << value0().x << ',' << value0().y;

        std::ofstream(temp_file_path()) << ss_vals.str() << std::endl;

        TestData read;

        {
            DbRecord record(temp_file_path(), key(), true);

            EXPECT(record.Load(legacy_id(), read));
        }

        EXPECT_EQUAL(value0(), read);
    }
};
#endif

class DbRecordOperationsTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        (void)std::ofstream(temp_file_path()); // To suppress warning in logs.

        TestData to_be_rewritten(7, 8);

        {
            DbRecord record(temp_file_path(), key());

            EXPECT(record.Store(id0(), to_be_rewritten));
            EXPECT(record.Store(id1(), to_be_rewritten));

            // Rewritting existing value with other.
            EXPECT(record.Store(id1(), value1()));

            // Rewritting existing value with same. In fact no DB manipulation should be performed
            // inside of store in such case.
            EXPECT(record.Store(id1(), value1()));
        }

        {
            DbRecord record(temp_file_path(), key());

            // Rewriting existing value to store it to file.
            EXPECT(record.Store(id0(), value0()));
        }

        {
            TestData read0, read1, read_missing, read_missing_cmp(read_missing);
            DbRecord record(temp_file_path(), key());

            // Loading by id not present in record should execute well but return false as nothing
            // was read.
            EXPECT(!record.Load(missing_id(), read_missing));

            // In such case value should not be changed.
            EXPECT_EQUAL(read_missing, read_missing_cmp);

            EXPECT(record.Load(id0(), read0));
            EXPECT(record.Load(id1(), read1));

            EXPECT_EQUAL(read0, value0());
            EXPECT_EQUAL(read1, value1());

            EXPECT(record.Remove(id0()));

            read0 = read_missing_cmp;

            EXPECT(!record.Load(id0(), read0));
            EXPECT(record.Load(id1(), read1));

            EXPECT_EQUAL(read0, read_missing_cmp);
            EXPECT_EQUAL(read1, value1());

            EXPECT(record.Remove(id0()));
        }

        {
            TestData read0, read1, read_missing_cmp(read0);
            DbRecord record(temp_file_path(), key());

            EXPECT(!record.Load(id0(), read0));
            EXPECT(record.Load(id1(), read1));

            EXPECT_EQUAL(read0, read_missing_cmp);
            EXPECT_EQUAL(read1, value1());
        }
    }
};

} // namespace tests
} // namespace miopen

int main()
{
    miopen::tests::DbRecordReadTest().Run();
    miopen::tests::DbRecordWriteTest().Run();
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    miopen::tests::DbRecordLegacyReadTest().Run();
#endif
    miopen::tests::DbRecordOperationsTest().Run();

    return 0;
}
