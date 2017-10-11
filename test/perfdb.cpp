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

    void LegacySerialize(std::ostream& s) const
    {
        Serialize(s);
        s << ",l";
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

    bool LegacyDeserialize(const std::string& s) { return Deserialize(s); }
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

struct TestValues
{
    TestData key{
        1, 2,
    };
    TestData value0{
        3, 4,
    };
    TestData value1{
        5, 6,
    };
    std::string id0 = "0";
    std::string id1 = "1";
};

class DbRecordTest
{
    public:
    virtual ~DbRecordTest() { std::remove(TempFilePath()); }

    protected:
    static const char* TempFilePath() { return "/tmp/dbread.test.temp.pdb"; }
};

class DbRecordReadTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        TestValues v;

        std::ostringstream ss_vals;
        ss_vals << v.key.x << ',' << v.key.y << '=' << v.id1 << ':' << v.value1.x << ','
                << v.value1.y << ';' << v.id0 << ':' << v.value0.x << ',' << v.value0.y;

        std::ofstream(TempFilePath()) << ss_vals.str() << std::endl;

        TestData read0, read1;

        {
            DbRecord record(TempFilePath(), v.key);

            EXPECT(record.Load(v.id0, read0));
            EXPECT(record.Load(v.id1, read1));
        }

        EXPECT_EQUAL(v.value0, read0);
        EXPECT_EQUAL(v.value1, read1);
    }
};

class DbRecordWriteTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        TestValues v;

        std::ostringstream ss_vals;
        ss_vals << v.key.x << ',' << v.key.y << '=' << v.id1 << ':' << v.value1.x << ','
                << v.value1.y << ';' << v.id0 << ':' << v.value0.x << ',' << v.value0.y;

        (void)std::ofstream(TempFilePath());

        {
            DbRecord record(TempFilePath(), v.key);

            EXPECT(record.Store(v.id0, v.value0));
            EXPECT(record.Store(v.id1, v.value1));
        }

        std::string read;

        EXPECT(std::getline(std::ifstream(TempFilePath()), read).good());
        EXPECT_EQUAL(read, ss_vals.str());
    }
};

class DbRecordLegacyReadTest : public DbRecordTest
{
    public:
    inline void Run()
    {
        TestValues v;

        std::ostringstream ss_vals;
        ss_vals << v.key.x << ',' << v.key.y << ",l " << v.value0.x << ',' << v.value0.y;

        std::ofstream(TempFilePath()) << ss_vals.str() << std::endl;

        TestData read;

        {
            DbRecord record(TempFilePath(), v.key, true);

            auto legacy_id = "ConvOclDirectFwd"; // const from db_record.cpp
            EXPECT(record.Load(legacy_id, read));
        }

        EXPECT_EQUAL(v.value0, read);
    }
};

} // namespace miopen
} // namespace tests

int main()
{
    miopen::tests::DbRecordReadTest().Run();
    miopen::tests::DbRecordWriteTest().Run();
    miopen::tests::DbRecordLegacyReadTest().Run();

    return 0;
}
