#include <cassert>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "miopen/db_record.hpp"
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
    virtual ~DbRecordTest() { std::remove(TempFilePath()); }

    protected:
    static const TestData key;
    static const TestData value0;
    static const TestData value1;
    static const char* const id0;
    static const char* const id1;
    static const char* const missing_id;
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
    static const char* const legacy_id;
#endif

    static const char* TempFilePath() { return "/tmp/dbread.test.temp.pdb"; }
};

const TestData DbRecordTest::key(1, 2);
const TestData DbRecordTest::value0(3, 4);
const TestData DbRecordTest::value1(5, 6);
const char* const DbRecordTest::id0        = "0";
const char* const DbRecordTest::id1        = "1";
const char* const DbRecordTest::missing_id = "2";
#if MIOPEN_PERFDB_CONV_LEGACY_SUPPORT
const char* const DbRecordTest::legacy_id  = "ConvOclDirectFwd"; // const from db_record.cpp
#endif

class DbRecordReadTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        std::ostringstream ss_vals;
        ss_vals << key.x << ',' << key.y << '=' << id1 << ':' << value1.x << ',' << value1.y << ';'
                << id0 << ':' << value0.x << ',' << value0.y;

        std::ofstream(TempFilePath()) << ss_vals.str() << std::endl;

        TestData read0, read1;

        {
            DbRecord record(TempFilePath(), key);

            EXPECT(record.Load(id0, read0));
            EXPECT(record.Load(id1, read1));
        }

        EXPECT_EQUAL(value0, read0);
        EXPECT_EQUAL(value1, read1);
    }
};

class DbRecordWriteTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        std::ostringstream ss_vals;
        ss_vals << key.x << ',' << key.y << '=' << id1 << ':' << value1.x << ',' << value1.y << ';'
                << id0 << ':' << value0.x << ',' << value0.y;

        (void)std::ofstream(TempFilePath());

        {
            DbRecord record(TempFilePath(), key);

            EXPECT(record.Store(id0, value0));
            EXPECT(record.Store(id1, value1));
        }

        std::string read;

        EXPECT(std::getline(std::ifstream(TempFilePath()), read).good());
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
        ss_vals << key.x << ',' << key.y << ",l " << value0.x << ',' << value0.y;

        std::ofstream(TempFilePath()) << ss_vals.str() << std::endl;

        TestData read;

        {
            DbRecord record(TempFilePath(), key, true);

            EXPECT(record.Load(legacy_id, read));
        }

        EXPECT_EQUAL(value0, read);
    }
};
#endif

class DbRecordOperationsTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        (void)std::ofstream(TempFilePath()); // To suppress warning in logs.

        TestData to_be_rewritten(7, 8);

        {
            DbRecord record(TempFilePath(), key);

            EXPECT(record.Store(id0, to_be_rewritten));
            EXPECT(record.Store(id1, to_be_rewritten));

            // Rewritting existing value with other.
            EXPECT(record.Store(id1, value1));

            // Rewritting existing value with same. In fact no DB manipulation should be performed
            // inside of store in such case.
            EXPECT(record.Store(id1, value1));
        }

        {
            DbRecord record(TempFilePath(), key);

            // Rewriting existing value to store it to file.
            EXPECT(record.Store(id0, value0));
        }

        TestData read0, read1, read_missing, read_missing_cmp(read_missing);

        {
            DbRecord record(TempFilePath(), key);

            // Loading by id not present in record should execute well but return false as nothing
            // was read.
            EXPECT(!record.Load(missing_id, read_missing));

            // In such case value should not be changed.
            EXPECT_EQUAL(read_missing, read_missing_cmp);

            EXPECT(record.Load(id0, read0));
            EXPECT(record.Load(id1, read1));
        }

        EXPECT_EQUAL(read0, value0);
        EXPECT_EQUAL(read1, value1);
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
