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

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/optional.hpp>

#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/temp_file.hpp>

#include "test.hpp"

namespace miopen {
namespace tests {

static boost::filesystem::path& exe_path()
{
    static boost::filesystem::path exe_path;
    return exe_path;
}

class Random
{
    public:
    Random(unsigned int seed = 0) : rng(seed) {}

    std::mt19937::result_type Next() { return dist(rng); }

    private:
    std::mt19937 rng;
    std::uniform_int_distribution<std::mt19937::result_type> dist;
};

struct TestData
{
    int x;
    int y;

    inline TestData() : x(Rnd().Next()), y(Rnd().Next()) {}

    inline TestData(int x_, int y_) : x(x_), y(y_) {}

    template <unsigned int seed>
    static inline TestData Seeded()
    {
        static Random rnd(seed);
        return {static_cast<int>(rnd.Next()), static_cast<int>(rnd.Next())};
    }

    inline void Serialize(std::ostream& s) const
    {
        static const auto sep = ',';
        s << x << sep << y;
    }

    inline bool Deserialize(const std::string& s)
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

    inline bool operator==(const TestData& other) const { return x == other.x && y == other.y; }

    private:
    static inline bool DeserializeField(std::istream& from, int* ret, char separator)
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

    static inline Random& Rnd()
    {
        static Random rnd;
        return rnd;
    }
};

std::ostream& operator<<(std::ostream& s, const TestData& td)
{
    s << "x: " << td.x << ", y: " << td.y;
    return s;
}

class DbTest
{
    public:
    DbTest() : _temp_file("miopen.tests.perfdb") {}
    virtual ~DbTest() {}

    protected:
    static const std::vector<std::pair<const char*, TestData>>& common_data()
    {
        static const std::vector<std::pair<const char*, TestData>> data
        {
            { id1(), value1() },
            { id0(), value0() },
        };

        return data;
    }

    inline void ResetDb() const
    {
        (void)std::ofstream(temp_file());
    }

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

    static const TestData& value2()
    {
        static const TestData data(7, 8);
        return data;
    }

    static const char* id0() { return "0"; }
    static const char* id1() { return "1"; }
    static const char* id2() { return "2"; }
    static const char* missing_id() { return "2"; }
    const TempFile& temp_file() const { return _temp_file; }

    template<class TKey, class TValue>
    static inline void RawWrite(const std::string& db_path, const TKey& key, const std::vector<std::pair<const char*, TValue>> values)
    {
        std::ostringstream ss_vals;
        ss_vals << key.x << ',' << key.y << '=';

        auto first = true;

        for (const auto& id_value : values)
        {
            if (!first)
                ss_vals << ";";

            first = false;
            ss_vals << id_value.first << ':' << id_value.second.x << ',' << id_value.second.y;
        }

        std::ofstream(db_path, std::ios::out | std::ios::ate) << ss_vals.str() << std::endl;
    }

    template<class TDb, class TKey, class TValue>
    inline void ValidateSingleEntry(TKey key, const std::vector<std::pair<const char*, TValue>> values, TDb db) const
    {
        boost::optional<DbRecord> record = db.FindRecord(key);

        EXPECT(record);

        for (const auto& id_value : values)
        {
            TValue read;
            EXPECT(record->GetValues(id_value.first, read));
            EXPECT_EQUAL(id_value.second, read);
        }
    }

    private:
    TempFile _temp_file;
};

class DbFindTest : public DbTest
{
    public:
    inline void Run() const
    {
        ResetDb();
        RawWrite(temp_file(), key(), common_data());

        Db db(temp_file());
        ValidateSingleEntry(key(), common_data(), db);

        TestData invalid_key(100, 200);
        auto record1 = db.FindRecord(invalid_key);
        EXPECT(!record1);
    }
};

class DbStoreTest : public DbTest
{
    public:
    inline void Run() const
    {
        ResetDb();
        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            Db db(temp_file());

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file()), read).good());

        ValidateSingleEntry(key(), common_data(), Db(temp_file()));
    }
};

class DbUpdateTest : public DbTest
{
    public:
    inline void Run() const
    {
        ResetDb();
        // Store record0 (key=id0:value0)
        DbRecord record0(key());
        EXPECT(record0.SetValues(id0(), value0()));

        {
            Db db(temp_file());

            EXPECT(db.StoreRecord(record0));
        }

        // Update with record1 (key=id1:value1)
        DbRecord record1(key());
        EXPECT(record1.SetValues(id1(), value1()));

        {
            Db db(temp_file());

            EXPECT(db.UpdateRecord(record1));
        }

        // Check record1 (key=id0:value0;id1:value1)
        TestData read0, read1;
        EXPECT(record1.GetValues(id0(), read0));
        EXPECT(record1.GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);

        // Check record that is stored in db (key=id0:value0;id1:value1)
        ValidateSingleEntry(key(), common_data(), Db(temp_file()));
    }
};

class DbRemoveTest : public DbTest
{
    public:
    inline void Run() const
    {
        ResetDb();
        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            Db db(temp_file());

            EXPECT(db.StoreRecord(record));
        }

        {
            Db db(temp_file());

            EXPECT(db.FindRecord(key()));
            EXPECT(db.RemoveRecord(key()));
            EXPECT(!db.FindRecord(key()));
        }
    }
};

class DbReadTest : public DbTest
{
    public:
    inline void Run() const
    {
        ResetDb();
        RawWrite(temp_file(), key(), common_data());
        ValidateSingleEntry(key(), common_data(), Db(temp_file()));
    }
};

class DbWriteTest : public DbTest
{
    public:
    inline void Run() const
    {
        ResetDb();

        {
            Db db(temp_file());

            EXPECT(db.Update(key(), id0(), value0()));
            EXPECT(db.Update(key(), id1(), value1()));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file()), read).good());

        ValidateSingleEntry(key(), common_data(), Db(temp_file()));
    }
};

class DbOperationsTest : public DbTest
{
    public:
    inline void Run() const
    {
        ResetDb();
        TestData to_be_rewritten(7, 8);

        {
            Db db(temp_file());

            EXPECT(db.Update(key(), id0(), to_be_rewritten));
            EXPECT(db.Update(key(), id1(), to_be_rewritten));

            // Rewritting existing value with other.
            EXPECT(db.Update(key(), id1(), value1()));

            // Rewritting existing value with same. In fact no DB manipulation should be performed
            // inside of store in such case.
            EXPECT(db.Update(key(), id1(), value1()));
        }

        {
            Db db(temp_file());

            // Rewriting existing value to store it to file.
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            TestData read0, read1, read_missing, read_missing_cmp(read_missing);
            Db db(temp_file());

            // Loading by id not present in record should execute well but return false as nothing
            // was read.
            EXPECT(!db.Load(key(), missing_id(), read_missing));

            // In such case value should not be changed.
            EXPECT_EQUAL(read_missing, read_missing_cmp);

            EXPECT(db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));

            EXPECT_EQUAL(read0, value0());
            EXPECT_EQUAL(read1, value1());

            EXPECT(db.Remove(key(), id0()));

            read0 = read_missing_cmp;

            EXPECT(!db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));

            EXPECT_EQUAL(read0, read_missing_cmp);
            EXPECT_EQUAL(read1, value1());

            EXPECT(!db.Remove(key(), id0()));
        }

        {
            TestData read0, read1, read_missing_cmp(read0);
            Db db(temp_file());

            EXPECT(!db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));

            EXPECT_EQUAL(read0, read_missing_cmp);
            EXPECT_EQUAL(read1, value1());
        }
    }
};

class DbParallelTest : public DbTest
{
    public:
    inline void Run() const
    {
        ResetDb();

        {
            Db db(temp_file());
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            Db db0(temp_file());
            Db db1(temp_file());

            auto r0 = db0.FindRecord(key());
            auto r1 = db1.FindRecord(key());

            EXPECT(r0);
            EXPECT(r1);

            EXPECT(r0->SetValues(id1(), value1()));
            EXPECT(r1->SetValues(id2(), value2()));

            EXPECT(db0.UpdateRecord(*r0));
            EXPECT(db1.UpdateRecord(*r1));
        }

        const std::vector<std::pair<const char*, TestData>> data
        {
            std::make_pair(id0(), value0()),
            std::make_pair(id1(), value1()),
            std::make_pair(id2(), value2()),
        };

        ValidateSingleEntry(key(), data, Db(temp_file()));
    }
};

class DBMultiThreadedTestWork
{
    public:
    static constexpr unsigned char threads_count   = 8;
    static constexpr unsigned int common_part_size = 128;
    static constexpr unsigned int unique_part_size = 128;
    static constexpr unsigned int ids_per_key      = 16;
    static constexpr unsigned int common_part_seed = 435345;

    static inline const std::array<TestData, common_part_size>& common_part()
    {
        static const std::array<TestData, common_part_size>& ref = common_part_init();
        return ref;
    }

    static inline void WorkItem(unsigned int id, const std::string& db_path)
    {
        CommonPart(db_path);
        UniquePart(id, db_path);
    }

    static inline void ValidateCommonPart(const std::string& db_path)
    {
        Db db(db_path);

        for(auto i = 0u; i < common_part_size; i++)
        {
            const auto key  = i / ids_per_key;
            const auto id   = i % ids_per_key;
            const auto data = common_part()[i];
            TestData read;

            EXPECT(db.Load(std::to_string(key), std::to_string(id), read));
            EXPECT_EQUAL(read, data);
        }
    }

    private:
    static inline void CommonPart(const std::string& db_path)
    {
        {
            Db db(db_path);
            CommonPartSection(0u, common_part_size / 2, [&db]() { return db; });
        }

        CommonPartSection(
            common_part_size / 2, common_part_size, [&db_path]() { return Db(db_path); });
    }

    template <class TDbGetter>
    static inline void
    CommonPartSection(unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        for(auto i = start; i < end; i++)
        {
            const auto key  = i / ids_per_key;
            const auto id   = i % ids_per_key;
            const auto data = common_part()[i];

            db_getter().Update(std::to_string(key), std::to_string(id), data);
        }
    }

    static inline void UniquePart(unsigned int id, const std::string& db_path)
    {
        Random rnd(123123 + id);

        {
            Db db(db_path);
            UniquePartSection(rnd, 0, unique_part_size / 2, [&db]() { return db; });
        }

        UniquePartSection(
            rnd, unique_part_size / 2, unique_part_size, [&db_path]() { return Db(db_path); });
    }

    template <class TDbGetter>
    static inline void
    UniquePartSection(Random& rnd, unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        for(auto i = start; i < end; i++)
        {
            auto key = LimitedRandom(rnd, common_part_size / ids_per_key + 2);
            auto id  = LimitedRandom(rnd, ids_per_key + 1);
            TestData data;

            db_getter().Update(std::to_string(key), std::to_string(id), data);
        }
    }

    static inline std::mt19937::result_type LimitedRandom(Random& rnd,
                                                          std::mt19937::result_type min)
    {
        std::mt19937::result_type key;

        do
            key = rnd.Next();
        while(key < min);

        return key;
    }

    static inline const std::array<TestData, common_part_size>& common_part_init()
    {
        static std::array<TestData, common_part_size> data;

        for(auto i  = 0u; i < common_part_size; i++)
            data[i] = TestData::Seeded<common_part_seed>();

        return data;
    }
};

class DbMultiThreadedTest : public DbTest
{
    public:
    inline void Run()
    {
        ResetDb();
        std::mutex mutex;
        std::vector<std::thread> threads;

        threads.reserve(DBMultiThreadedTestWork::threads_count);

        {
            std::unique_lock<std::mutex> lock(mutex);
            auto id = 0;

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
                threads.emplace_back([this, &mutex, &id]() {
                    (void)std::unique_lock<std::mutex>(mutex);
                    DBMultiThreadedTestWork::WorkItem(id++, temp_file());
                });
        }

        for(auto& thread : threads)
            thread.join();

        DBMultiThreadedTestWork::ValidateCommonPart(temp_file());
    }
};

class DbMultiProcessTest : public DbTest
{
    public:
    static constexpr const char* arg = "-mp-test-child";

    inline void Run() const
    {
        ResetDb();
        std::vector<FILE*> children(DBMultiThreadedTestWork::threads_count);
        const auto lock_file_path = LockFilePath(temp_file());

        {
            auto file_lock = LockFileDispatcher::Get(lock_file_path.c_str());
            boost::interprocess::scoped_lock<LockFile> lock(file_lock);

            auto id = 0;

            for(auto& child : children)
            {
                const auto command = exe_path().string() + " " + arg + " " + std::to_string(id++) +
                                     " " + temp_file().Path();
                child = popen(command.c_str(), "w");
            }
        }

        for(auto child : children)
        {
            auto status          = pclose(child);
            const auto exit_code = WEXITSTATUS(status);

            EXPECT_EQUAL(exit_code, 0);
        }

        std::remove(lock_file_path.c_str());
        DBMultiThreadedTestWork::ValidateCommonPart(temp_file());
    }

    static inline void WorkItem(unsigned int id, const std::string& db_path)
    {
        {
            auto file_lock = LockFileDispatcher::Get(LockFilePath(db_path).c_str());
            boost::interprocess::sharable_lock<LockFile> lock(file_lock);
        }

        DBMultiThreadedTestWork::WorkItem(id, db_path);
    }

    private:
    static std::string LockFilePath(const std::string& db_path) { return db_path + ".test.lock"; }
};

class DbMultiFileTest : public DbTest
{
protected:
    const std::string& user_db_path() const { return _user_db_path; }

    inline void ResetDb() const
    {
        DbTest::ResetDb();
        (void)std::ofstream(user_db_path());
    }

private:
    const std::string _user_db_path = temp_file().Path() + ".user";
};

class DbMultiFileReadTest : public DbMultiFileTest
{
    public:
    inline void Run() const
    {
        ResetDb();
        MergedAndMissing();

        ResetDb();
        ReadUser();

        ResetDb();
        ReadInstalled();

        ResetDb();
        ReadConflict();
    }

private:
    static const std::vector<std::pair<const char*, TestData>>& single_item_data()
    {
        static const std::vector<std::pair<const char*, TestData>> data
        {
            { id0(), value0() },
        };

        return data;
    }

    inline void MergedAndMissing() const
    {
        RawWrite(temp_file(), key(), common_data());

        MultiFileDb db(temp_file(), user_db_path());
        ValidateSingleEntry(key(), common_data(), db);

        TestData invalid_key(100, 200);
        auto record1 = db.FindRecord(invalid_key);
        EXPECT(!record1);
    }

    inline void ReadUser() const
    {
        RawWrite(user_db_path(), key(), single_item_data());
        ValidateSingleEntry(key(), single_item_data(), MultiFileDb(temp_file(), user_db_path()));
    }

    inline void ReadInstalled() const
    {
        RawWrite(temp_file(), key(), single_item_data());
        ValidateSingleEntry(key(), single_item_data(), MultiFileDb(temp_file(), user_db_path()));
    }

    inline void ReadConflict() const
    {
        RawWrite(temp_file(), key(), single_item_data());
        ReadUser();
    }

};

class DbMultiFileWriteTest : public DbMultiFileTest
{
    public:
    inline void Run() const
    {
        ResetDb();

        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            MultiFileDb db(temp_file(), user_db_path());

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(!std::getline(std::ifstream(temp_file()), read).good());
        EXPECT(std::getline(std::ifstream(user_db_path()), read).good());

        ValidateSingleEntry(key(), common_data(), MultiFileDb(temp_file(), user_db_path()));
    }
};

} // namespace tests
} // namespace miopen

int main(int argsn, char** argsc)
{
    if(argsn >= 4 && argsc[1] == std::string(miopen::tests::DbMultiProcessTest::arg))
    {
        miopen::tests::DbMultiProcessTest::WorkItem(strtol(argsc[2], nullptr, 10), argsc[3]);
        return 0;
    }

    miopen::tests::exe_path() =
        boost::filesystem::system_complete(boost::filesystem::path(argsc[0]));

    miopen::tests::DbFindTest().Run();
    miopen::tests::DbStoreTest().Run();
    miopen::tests::DbUpdateTest().Run();
    miopen::tests::DbRemoveTest().Run();
    miopen::tests::DbReadTest().Run();
    miopen::tests::DbWriteTest().Run();
    miopen::tests::DbOperationsTest().Run();
    miopen::tests::DbParallelTest().Run();
    miopen::tests::DbMultiThreadedTest().Run();
    miopen::tests::DbMultiProcessTest().Run();
    miopen::tests::DbMultiFileReadTest().Run();
    miopen::tests::DbMultiFileWriteTest().Run();

    return 0;
}
