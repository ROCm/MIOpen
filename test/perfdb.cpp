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
#include <atomic>

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

static boost::optional<std::string>& thread_logs_root()
{
    static boost::optional<std::string> path = boost::none;
    return path;
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
    static const char* missing_id() { return "3"; }
    const TempFile& temp_file_path() const { return _temp_file; }

    private:
    TempFile _temp_file;
};

class DbFindTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for reading premade file by FindRecord..." << std::endl;

        std::ostringstream ss_vals;
        ss_vals << key().x << ',' << key().y << '=' << id1() << ':' << value1().x << ','
                << value1().y << ';' << id0() << ':' << value0().x << ',' << value0().y;

        std::ofstream(temp_file_path()) << ss_vals.str() << std::endl;

        boost::optional<DbRecord> record0, record1;
        TestData read0, read1;
        TestData invalid_key(100, 200);

        {
            Db db(temp_file_path());

            record0 = db.FindRecord(key());
            record1 = db.FindRecord(invalid_key);
        }

        EXPECT(record0);
        EXPECT(record0->GetValues(id0(), read0));
        EXPECT(record0->GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
        EXPECT(!record1);
    }
};

class DbStoreTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for reading stored data..." << std::endl;

        (void)std::ofstream(temp_file_path());

        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            Db db(temp_file_path());

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file_path()), read).good());

        boost::optional<DbRecord> record_read;
        TestData read0, read1;

        {
            Db db(temp_file_path());

            record_read = db.FindRecord(key());
        }

        EXPECT(record_read);
        EXPECT(record_read->GetValues(id0(), read0));
        EXPECT(record_read->GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
    }
};

class DbUpdateTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for updating existing records..." << std::endl;

        (void)std::ofstream(temp_file_path());

        // Store record0 (key=id0:value0)
        DbRecord record0(key());
        EXPECT(record0.SetValues(id0(), value0()));

        {
            Db db(temp_file_path());

            EXPECT(db.StoreRecord(record0));
        }

        // Update with record1 (key=id1:value1)
        DbRecord record1(key());
        EXPECT(record1.SetValues(id1(), value1()));

        {
            Db db(temp_file_path());

            EXPECT(db.UpdateRecord(record1));
        }

        // Check record1 (key=id0:value0;id1:value1)
        TestData read0, read1;
        EXPECT(record1.GetValues(id0(), read0));
        EXPECT(record1.GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);

        // Check record that is stored in db (key=id0:value0;id1:value1)
        boost::optional<DbRecord> record_read;
        {
            Db db(temp_file_path());

            record_read = db.FindRecord(key());
        }

        EXPECT(record_read);
        EXPECT(record_read->GetValues(id0(), read0));
        EXPECT(record_read->GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
    }
};

class DbRemoveTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for removing records..." << std::endl;

        (void)std::ofstream(temp_file_path());

        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            Db db(temp_file_path());

            EXPECT(db.StoreRecord(record));
        }

        {
            Db db(temp_file_path());

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
        std::cout << "Testing db for reading premade file by Load..." << std::endl;

        std::ostringstream ss_vals;
        ss_vals << key().x << ',' << key().y << '=' << id1() << ':' << value1().x << ','
                << value1().y << ';' << id0() << ':' << value0().x << ',' << value0().y;

        std::ofstream(temp_file_path()) << ss_vals.str() << std::endl;

        TestData read0, read1;

        {
            Db db(temp_file_path());

            EXPECT(db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));
        }

        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
    }
};

class DbWriteTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for storing unexistent records by update..." << std::endl;

        (void)std::ofstream(temp_file_path());

        {
            Db db(temp_file_path());

            EXPECT(db.Update(key(), id0(), value0()));
            EXPECT(db.Update(key(), id1(), value1()));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file_path()), read).good());

        TestData read0, read1;

        {
            Db db(temp_file_path());

            EXPECT(db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));
        }

        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);
    }
};

class DbOperationsTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing different db operations db..." << std::endl;

        (void)std::ofstream(temp_file_path()); // To suppress warning in logs.

        TestData to_be_rewritten(7, 8);

        {
            Db db(temp_file_path());

            EXPECT(db.Update(key(), id0(), to_be_rewritten));
            EXPECT(db.Update(key(), id1(), to_be_rewritten));

            // Rewritting existing value with other.
            EXPECT(db.Update(key(), id1(), value1()));

            // Rewritting existing value with same. In fact no DB manipulation should be performed
            // inside of store in such case.
            EXPECT(db.Update(key(), id1(), value1()));
        }

        {
            Db db(temp_file_path());

            // Rewriting existing value to store it to file.
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            TestData read0, read1, read_missing, read_missing_cmp(read_missing);
            Db db(temp_file_path());

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
            Db db(temp_file_path());

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
        std::cout << "Testing db for using two objects targeting one file existing in one scope..."
                  << std::endl;

        {
            Db db(temp_file_path());
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            Db db0(temp_file_path());
            Db db1(temp_file_path());

            auto r0 = db0.FindRecord(key());
            auto r1 = db0.FindRecord(key());

            EXPECT(r0);
            EXPECT(r1);

            EXPECT(r0->SetValues(id1(), value1()));
            EXPECT(r1->SetValues(id2(), value2()));

            EXPECT(db0.UpdateRecord(*r0));
            EXPECT(db1.UpdateRecord(*r1));
        }

        {
            Db db(temp_file_path());
            TestData read1, read2;

            EXPECT(db.Load(key(), id1(), read1));
            EXPECT(db.Load(key(), id2(), read2));

            EXPECT_EQUAL(read1, value1());
            EXPECT_EQUAL(read2, value2());
        }
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

    static inline void
    WorkItem(unsigned int id, const std::string& db_path, const std::string& log_postfix)
    {
        std::ofstream log;
        std::ofstream log_err;
        std::streambuf *cout_buf = nullptr, *cerr_buf = nullptr;

        if(thread_logs_root())
        {
            const auto out_path =
                *thread_logs_root() + "/thread-" + std::to_string(id) + "_" + log_postfix + ".log";
            const auto err_path = *thread_logs_root() + "/thread-" + std::to_string(id) + "_" +
                                  log_postfix + "-err.log";

            std::remove(out_path.c_str());
            std::remove(err_path.c_str());

            log.open(out_path);
            log_err.open(err_path);
            cout_buf = std::cout.rdbuf();
            cerr_buf = std::cerr.rdbuf();
            std::cout.rdbuf(log.rdbuf());
            std::cerr.rdbuf(log_err.rdbuf());
        }

        CommonPart(db_path);
        UniquePart(id, db_path);

        if(thread_logs_root())
        {
            std::cout.rdbuf(cout_buf);
            std::cerr.rdbuf(cerr_buf);
        }
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
        std::cout << "Common part. Section with common db instance." << std::endl;
        {
            Db db(db_path);
            CommonPartSection(0u, common_part_size / 2, [&db]() { return db; });
        }

        std::cout << "Common part. Section with separate db instances." << std::endl;
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

        std::cout << "Unique part. Section with common db instance." << std::endl;
        {
            Db db(db_path);
            UniquePartSection(rnd, 0, unique_part_size / 2, [&db]() { return db; });
        }

        std::cout << "Unique part. Section with separate db instances." << std::endl;
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

        for(auto i = 0u; i < common_part_size; i++)
            data[i] = TestData::Seeded<common_part_seed>();

        return data;
    }
};

class DbMultiThreadedTest : public DbTest
{
    public:
    inline void Run()
    {
        std::cout << "Testing db for multithreaded access..." << std::endl;

        std::mutex mutex;
        std::vector<std::thread> threads;

        std::cout << "Launching test threads..." << std::endl;
        threads.reserve(DBMultiThreadedTestWork::threads_count);

        {
            std::string p = temp_file_path();
            std::unique_lock<std::mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
                threads.emplace_back([p, &mutex, i]() {
                    (void)std::unique_lock<std::mutex>(mutex);
                    DBMultiThreadedTestWork::WorkItem(i, p, "mt");
                });
        }

        std::cout << "Waiting for test threads..." << std::endl;
        for(auto& thread : threads)
            thread.join();

        std::cout << "Validation results..." << std::endl;
        DBMultiThreadedTestWork::ValidateCommonPart(temp_file_path());
    }
};

class DbMultiProcessTest : public DbTest
{
    public:
    static constexpr const char* arg = "-mp-test-child";

    inline void Run() const
    {
        std::cout << "Testing db for multiprocess access..." << std::endl;

        std::vector<FILE*> children(DBMultiThreadedTestWork::threads_count);
        const auto lock_file_path = LockFilePath(temp_file_path());

        std::cout << "Launching test processes..." << std::endl;
        {
            auto& file_lock = LockFile::Get(lock_file_path.c_str());
            std::shared_lock<LockFile> lock(file_lock);

            auto id = 0;

            for(auto& child : children)
            {
                auto command = exe_path().string() + " " + arg + " " + std::to_string(id++) + " " +
                               temp_file_path().Path();

                if(thread_logs_root())
                    command += " " + *thread_logs_root();

                child = popen(command.c_str(), "w");
            }
        }

        std::cout << "Waiting for test processes..." << std::endl;
        for(auto child : children)
        {
            auto status          = pclose(child);
            const auto exit_code = WEXITSTATUS(status);

            EXPECT_EQUAL(exit_code, 0);
        }

        std::remove(lock_file_path.c_str());
        std::cout << "Validation results..." << std::endl;
        DBMultiThreadedTestWork::ValidateCommonPart(temp_file_path());
    }

    static inline void WorkItem(unsigned int id, const std::string& db_path)
    {
        {
            auto& file_lock = LockFile::Get(LockFilePath(db_path).c_str());
            std::lock_guard<LockFile> lock(file_lock);
        }

        DBMultiThreadedTestWork::WorkItem(id, db_path, "mp");
    }

    private:
    static std::string LockFilePath(const std::string& db_path) { return db_path + ".test.lock"; }
};

} // namespace tests
} // namespace miopen

int main(int argsn, char** argsc)
{
    if(argsn >= 3 && argsc[1] == std::string("-thread-logs-root"))
        miopen::tests::thread_logs_root() = argsc[2];

    if(argsn >= 4 && argsc[1] == std::string(miopen::tests::DbMultiProcessTest::arg))
    {
        if(argsn >= 5)
            miopen::tests::thread_logs_root() = argsc[4];

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

    return 0;
}
