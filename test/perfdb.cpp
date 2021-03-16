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

#include "test.hpp"
#include "driver.hpp"

#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/temp_file.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/optional.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace miopen {
namespace tests {

static boost::filesystem::path& exe_path()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static boost::filesystem::path exe_path;
    return exe_path;
}

static boost::optional<std::string>& thread_logs_root()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static boost::optional<std::string> path(boost::none);
    return path;
}

static bool& full_set()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool full_set = false;
    return full_set;
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

    struct NoInit
    {
    };

    TestData(NoInit) : x(0), y(0) {}
    TestData(Random& rnd) : x(rnd.Next()), y(rnd.Next()) {}
    TestData() : x(Rnd().Next()), y(Rnd().Next()) {}
    TestData(int x_, int y_) : x(x_), y(y_) {}

    template <unsigned int seed>
    static TestData Seeded()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static Random rnd(seed);
        return {static_cast<int>(rnd.Next()), static_cast<int>(rnd.Next())};
    }

    void Serialize(std::ostream& s) const
    {
        static const auto sep = ',';
        s << x << sep << y;
    }

    bool Deserialize(const std::string& s)
    {
        static const auto sep = ',';
        TestData t(NoInit{});
        std::istringstream ss(s);

        const auto success = DeserializeField(ss, &t.x, sep) && DeserializeField(ss, &t.y, sep);

        if(!success)
            return false;

        *this = t;
        return true;
    }

    bool operator==(const TestData& other) const { return x == other.x && y == other.y; }

    private:
    static bool DeserializeField(std::istream& from, int* ret, char separator)
    {
        std::string part;

        if(!std::getline(from, part, separator))
            return false;

        const auto start = part.c_str();
        char* end;
        const auto value = std::strtol(start, &end, 10);

        if(start == end)
            return false;

        *ret = value;
        return true;
    }

    static Random& Rnd()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
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
    DbTest() : temp_file("miopen.tests.perfdb") {}

    virtual ~DbTest() { std::remove(LockFilePath(temp_file.Path()).c_str()); }

    protected:
    TempFile temp_file;

    static const std::array<std::pair<const std::string, TestData>, 2>& common_data()
    {
        static const std::array<std::pair<const std::string, TestData>, 2> data{{
            {id1(), value1()}, {id0(), value0()},
        }};

        return data;
    }

    void ResetDb() const { (void)std::ofstream(temp_file); }

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

    static const std::string& id0()
    {
        static const std::string id0_ = "0";
        return id0_;
    }
    static const std::string& id1()
    {
        static const std::string id1_ = "1";
        return id1_;
    }
    static const std::string& id2()
    {
        static const std::string id2_ = "2";
        return id2_;
    }
    static const std::string& missing_id()
    {
        static const std::string missing_id_ = "3";
        return missing_id_;
    }

    template <class TKey, class TValue, size_t count>
    static void RawWrite(const std::string& db_path,
                         const TKey& key,
                         const std::array<std::pair<const std::string, TValue>, count> values)
    {
        std::ostringstream ss_vals;
        ss_vals << key.x << ',' << key.y << '=';

        auto first = true;

        for(const auto& id_value : values)
        {
            if(!first)
                ss_vals << ";";

            first = false;
            ss_vals << id_value.first << ':' << id_value.second.x << ',' << id_value.second.y;
        }

        std::ofstream(db_path, std::ios::out | std::ios::ate) << ss_vals.str() << std::endl;
    }

    template <class TDb, class TKey, class TValue, size_t count>
    static void ValidateSingleEntry(
        TKey key, const std::array<std::pair<const std::string, TValue>, count> values, TDb db)
    {
        boost::optional<DbRecord> record = db.FindRecord(key);

        EXPECT(record);

        for(const auto& id_value : values)
        {
            TValue read;
            EXPECT(record->GetValues(id_value.first, read));
            EXPECT_EQUAL(id_value.second, read);
        }
    }
};

class DbFindTest : public DbTest
{
    public:
    void Run() const
    {
        std::cout << "Testing db for reading premade file by FindRecord..." << std::endl;

        ResetDb();
        RawWrite(temp_file, key(), common_data());

        PlainTextDb db(temp_file);
        ValidateSingleEntry(key(), common_data(), db);

        const TestData invalid_key(100, 200);
        auto record1 = db.FindRecord(invalid_key);
        EXPECT(!record1);
    }
};

class DbStoreTest : public DbTest
{
    public:
    void Run() const
    {
        std::cout << "Testing db for reading stored data..." << std::endl;

        ResetDb();
        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            PlainTextDb db(temp_file);

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file), read).good());

        ValidateSingleEntry(key(), common_data(), PlainTextDb(temp_file));
    }
};

class DbUpdateTest : public DbTest
{
    public:
    void Run() const
    {
        std::cout << "Testing db for updating existing records..." << std::endl;

        ResetDb();
        // Store record0 (key=id0:value0)
        DbRecord record0(key());
        EXPECT(record0.SetValues(id0(), value0()));

        {
            PlainTextDb db(temp_file);

            EXPECT(db.StoreRecord(record0));
        }

        // Update with record1 (key=id1:value1)
        DbRecord record1(key());
        EXPECT(record1.SetValues(id1(), value1()));

        {
            PlainTextDb db(temp_file);

            EXPECT(db.UpdateRecord(record1));
        }

        // Check record1 (key=id0:value0;id1:value1)
        TestData read0, read1;
        EXPECT(record1.GetValues(id0(), read0));
        EXPECT(record1.GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);

        // Check record that is stored in db (key=id0:value0;id1:value1)
        ValidateSingleEntry(key(), common_data(), PlainTextDb(temp_file));
    }
};

class DbRemoveTest : public DbTest
{
    public:
    void Run() const
    {
        std::cout << "Testing db for removing records..." << std::endl;

        ResetDb();
        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            PlainTextDb db(temp_file);

            EXPECT(db.StoreRecord(record));
        }

        {
            PlainTextDb db(temp_file);

            EXPECT(db.FindRecord(key()));
            EXPECT(db.RemoveRecord(key()));
            EXPECT(!db.FindRecord(key()));
        }
    }
};

class DbReadTest : public DbTest
{
    public:
    void Run() const
    {
        std::cout << "Testing db for reading premade file by Load..." << std::endl;

        ResetDb();
        RawWrite(temp_file, key(), common_data());
        ValidateSingleEntry(key(), common_data(), PlainTextDb(temp_file));
    }
};

class DbWriteTest : public DbTest
{
    public:
    void Run() const
    {
        std::cout << "Testing db for storing unexistent records by update..." << std::endl;

        ResetDb();

        {
            PlainTextDb db(temp_file);

            EXPECT(db.Update(key(), id0(), value0()));
            EXPECT(db.Update(key(), id1(), value1()));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file), read).good());

        ValidateSingleEntry(key(), common_data(), PlainTextDb(temp_file));
    }
};

class DbOperationsTest : public DbTest
{
    public:
    void Run() const
    {
        std::cout << "Testing different db operations db..." << std::endl;

        ResetDb();
        const TestData to_be_rewritten(7, 8);

        {
            PlainTextDb db(temp_file);

            EXPECT(db.Update(key(), id0(), to_be_rewritten));
            EXPECT(db.Update(key(), id1(), to_be_rewritten));

            // Rewritting existing value with other.
            EXPECT(db.Update(key(), id1(), value1()));

            // Rewritting existing value with same. In fact no DB manipulation should be performed
            // inside of store in such case.
            EXPECT(db.Update(key(), id1(), value1()));
        }

        {
            PlainTextDb db(temp_file);

            // Rewriting existing value to store it to file.
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            TestData read0, read1, read_missing;
            const auto read_missing_cmp(read_missing);
            PlainTextDb db(temp_file);

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
            TestData read0, read1;
            const auto read_missing_cmp(read0);
            PlainTextDb db(temp_file);

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
    void Run() const
    {
        std::cout << "Testing db for using two objects targeting one file existing in one scope..."
                  << std::endl;

        ResetDb();

        {
            PlainTextDb db(temp_file);
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            PlainTextDb db0(temp_file);
            PlainTextDb db1(temp_file);

            auto r0 = db0.FindRecord(key());
            auto r1 = db1.FindRecord(key());

            EXPECT(r0);
            EXPECT(r1);

            EXPECT(r0->SetValues(id1(), value1()));
            EXPECT(r1->SetValues(id2(), value2()));

            EXPECT(db0.UpdateRecord(*r0));
            EXPECT(db1.UpdateRecord(*r1));
        }

        const std::array<std::pair<const std::string, TestData>, 3> data{{
            {id0(), value0()}, {id1(), value1()}, {id2(), value2()},
        }};

        ValidateSingleEntry(key(), data, PlainTextDb(temp_file));
    }
};

class DBMultiThreadedTestWork
{
    public:
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static unsigned int threads_count;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static unsigned int common_part_size;
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static unsigned int unique_part_size;
    static constexpr unsigned int ids_per_key      = 16;
    static constexpr unsigned int common_part_seed = 435345;

    static const std::vector<TestData>& common_part()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        static const auto& ref = common_part_init();
        return ref;
    }

    static void Initialize() { (void)common_part(); }

    template <class TDbConstructor>
    static void
    WorkItem(unsigned int id, const TDbConstructor& db_constructor, const std::string& log_postfix)
    {
        RedirectLogs(id, log_postfix, [id, &db_constructor]() {
            CommonPart(db_constructor);
            UniquePart(id, db_constructor);
        });
    }

    template <class TDbConstructor>
    static void ReadWorkItem(unsigned int id,
                             const TDbConstructor& db_constructor,
                             const std::string& log_postfix)
    {
        RedirectLogs(id, log_postfix, [&db_constructor]() { ReadCommonPart(db_constructor); });
    }

    template <class TDbConstructor>
    static void FillForReading(const TDbConstructor& db_constructor)
    {
        auto db = db_constructor();
        CommonPartSection(0u, common_part_size, [&db]() { return db; });
    }

    template <class TDbConstructor>
    static void ValidateCommonPart(const TDbConstructor& db_constructor)
    {
        auto db       = db_constructor();
        const auto cp = common_part();

        for(auto i = 0u; i < common_part_size; i++)
        {
            const auto key  = std::to_string(i / ids_per_key);
            const auto id   = std::to_string(i % ids_per_key);
            const auto data = cp[i];
            TestData read(TestData::NoInit{});

            EXPECT(db.Load(key, id, read));
            EXPECT_EQUAL(read, data);
        }
    }

    private:
    template <class TWorker>
    static void RedirectLogs(unsigned int id, const std::string& log_postfix, const TWorker& worker)
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

        worker();

        if(thread_logs_root())
        {
            std::cout.rdbuf(cout_buf);
            std::cerr.rdbuf(cerr_buf);
        }
    }

    template <class TDbConstructor>
    static void ReadCommonPart(const TDbConstructor& db_constructor)
    {
        std::cout << "Common part. Section with common db instance." << std::endl;
        {
            auto db = db_constructor();
            ReadCommonPartSection(0u, common_part_size / 2, [&db]() { return db; });
        }

        std::cout << "Common part. Section with separate db instances." << std::endl;
        ReadCommonPartSection(common_part_size / 2, common_part_size, [&db_constructor]() {
            return db_constructor();
        });
    }

    template <class TDbGetter>
    static void
    ReadCommonPartSection(unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        const auto cp = common_part();

        for(auto i = start; i < end; i++)
        {
            const auto key  = std::to_string(i / ids_per_key);
            const auto id   = std::to_string(i % ids_per_key);
            const auto data = cp[i];
            TestData read(TestData::NoInit{});

            EXPECT(db_getter().Load(key, id, read));
            EXPECT_EQUAL(read, data);
        }
    }

    template <class TDbConstructor>
    static void CommonPart(const TDbConstructor& db_constructor)
    {
        std::cout << "Common part. Section with common db instance." << std::endl;
        {
            auto db = db_constructor();
            CommonPartSection(0u, common_part_size / 2, [&db]() { return db; });
        }

        std::cout << "Common part. Section with separate db instances." << std::endl;
        CommonPartSection(common_part_size / 2, common_part_size, [&db_constructor]() {
            return db_constructor();
        });
    }

    template <class TDbGetter>
    static void CommonPartSection(unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        const auto cp = common_part();

        for(auto i = start; i < end; i++)
        {
            const auto key  = std::to_string(i / ids_per_key);
            const auto id   = std::to_string(i % ids_per_key);
            const auto data = cp[i];

            db_getter().Update(key, id, data);
        }
    }

    template <class TDbConstructor>
    static void UniquePart(unsigned int id, const TDbConstructor& db_constructor)
    {
        Random rnd(123123 + id);

        std::cout << "Unique part. Section with common db instance." << std::endl;
        {
            auto db = db_constructor();
            UniquePartSection(rnd, 0, unique_part_size / 2, [&db]() { return db; });
        }

        std::cout << "Unique part. Section with separate db instances." << std::endl;
        UniquePartSection(rnd, unique_part_size / 2, unique_part_size, [&db_constructor]() {
            return db_constructor();
        });
    }

    template <class TDbGetter>
    static void
    UniquePartSection(Random& rnd, unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        for(auto i = start; i < end; i++)
        {
            auto key = LimitedRandom(rnd, common_part_size / ids_per_key + 2);
            auto id  = LimitedRandom(rnd, ids_per_key + 1);
            TestData data(rnd);

            db_getter().Update(std::to_string(key), std::to_string(id), data);
        }
    }

    static std::mt19937::result_type LimitedRandom(Random& rnd, std::mt19937::result_type min)
    {
        std::mt19937::result_type key;

        do
            key = rnd.Next();
        while(key < min);

        return key;
    }

    static const std::vector<TestData>& common_part_init()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static std::vector<TestData> data(common_part_size, TestData{TestData::NoInit{}});

        for(auto i  = 0u; i < common_part_size; i++)
            data[i] = TestData::Seeded<common_part_seed>();

        return data;
    }
};
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
unsigned int DBMultiThreadedTestWork::threads_count = 16;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
unsigned int DBMultiThreadedTestWork::common_part_size = 32;
// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
unsigned int DBMultiThreadedTestWork::unique_part_size = 32;

class DbMultiThreadedTest : public DbTest
{
    public:
    static constexpr const char* logs_path_arg = "thread-logs-root";

    void Run() const
    {
        std::cout << "Testing db for multithreaded write access..." << std::endl;

        ResetDb();
        std::mutex mutex;
        std::vector<std::thread> threads;

        std::cout << "Initializing test data..." << std::endl;
        DBMultiThreadedTestWork::Initialize();

        std::cout << "Launching test threads..." << std::endl;
        threads.reserve(DBMultiThreadedTestWork::threads_count);
        const std::string p = temp_file;
        const auto c        = [&p]() { return PlainTextDb(p); };

        {
            std::unique_lock<std::mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
                threads.emplace_back([c, &mutex, i]() {
                    (void)std::unique_lock<std::mutex>(mutex);
                    DBMultiThreadedTestWork::WorkItem(i, c, "mt");
                });
        }

        std::cout << "Waiting for test threads..." << std::endl;
        for(auto& thread : threads)
            thread.join();

        std::cout << "Validating results..." << std::endl;
        DBMultiThreadedTestWork::ValidateCommonPart(c);
        std::cout << "Validation passed..." << std::endl;
    }
};

class DbMultiThreadedReadTest : public DbTest
{
    public:
    void Run() const
    {
        std::cout << "Testing db for multithreaded read access..." << std::endl;

        std::mutex mutex;
        std::vector<std::thread> threads;

        std::cout << "Initializing test data..." << std::endl;
        const std::string p = temp_file;
        const auto c        = [&p]() { return PlainTextDb(p); };
        ResetDb();
        DBMultiThreadedTestWork::FillForReading(c);

        std::cout << "Launching test threads..." << std::endl;
        threads.reserve(DBMultiThreadedTestWork::threads_count);

        {
            std::unique_lock<std::mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
                threads.emplace_back([c, &mutex, i]() {
                    (void)std::unique_lock<std::mutex>(mutex);
                    DBMultiThreadedTestWork::ReadWorkItem(i, c, "mt");
                });
        }

        std::cout << "Waiting for test threads..." << std::endl;
        for(auto& thread : threads)
            thread.join();
    }
};

class DbMultiProcessTest : public DbTest
{
    public:
    static constexpr const char* write_arg = "mp-test-child-write";
    static constexpr const char* id_arg    = "mp-test-child";
    static constexpr const char* path_arg  = "mp-test-child-path";

    void Run() const
    {
        std::cout << "Testing db for multiprocess write access..." << std::endl;

        ResetDb();
        std::vector<FILE*> children(DBMultiThreadedTestWork::threads_count);
        const auto lock_file_path = LockFilePath(temp_file);

        std::cout << "Initializing test data..." << std::endl;
        DBMultiThreadedTestWork::Initialize();

        std::cout << "Launching test processes..." << std::endl;
        {
            auto& file_lock = LockFile::Get(lock_file_path.c_str());
            std::shared_lock<LockFile> lock(file_lock);

            auto id = 0;

            for(auto& child : children)
            {
                auto command = exe_path().string() + " --" + write_arg + " --" + id_arg + " " +
                               std::to_string(id++) + " --" + path_arg + " " + temp_file.Path();

                if(thread_logs_root())
                    command += std::string(" --") + DbMultiThreadedTest::logs_path_arg + " " +
                               *thread_logs_root();

                if(full_set())
                    command += " --all";

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

        const std::string p = temp_file;
        const auto c        = [&p]() { return PlainTextDb(p); };

        std::cout << "Validating results..." << std::endl;
        DBMultiThreadedTestWork::ValidateCommonPart(c);
        std::cout << "Validation passed..." << std::endl;
    }

    static void WorkItem(unsigned int id, const std::string& db_path, bool write)
    {
        {
            auto& file_lock = LockFile::Get(LockFilePath(db_path).c_str());
            std::lock_guard<LockFile> lock(file_lock);
        }

        const auto c = [&db_path]() { return PlainTextDb(db_path); };

        if(write)
            DBMultiThreadedTestWork::WorkItem(id, c, "mp");
        else
            DBMultiThreadedTestWork::ReadWorkItem(id, c, "mp");
    }

    private:
    static std::string LockFilePath(const std::string& db_path) { return db_path + ".test.lock"; }
};

class DbMultiProcessReadTest : public DbTest
{
    public:
    void Run() const
    {
        std::cout << "Testing db for multiprocess read access..." << std::endl;

        std::vector<FILE*> children(DBMultiThreadedTestWork::threads_count);
        const auto lock_file_path = LockFilePath(temp_file);

        std::cout << "Initializing test data..." << std::endl;
        std::string p = temp_file;
        const auto c  = [&p]() { return PlainTextDb(p); };
        DBMultiThreadedTestWork::FillForReading(c);

        std::cout << "Launching test processes..." << std::endl;
        {
            auto& file_lock = LockFile::Get(lock_file_path.c_str());
            std::shared_lock<LockFile> lock(file_lock);

            auto id = 0;

            for(auto& child : children)
            {
                auto command = exe_path().string() + " --" + DbMultiProcessTest::id_arg + " " +
                               std::to_string(id++) + " --" + DbMultiProcessTest::path_arg + " " +
                               p;

                if(thread_logs_root())
                    command += std::string(" --") + DbMultiThreadedTest::logs_path_arg + " " +
                               *thread_logs_root();

                if(full_set())
                    command += " --all";

                std::cout << command << std::endl;
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
    }

    static void WorkItem(unsigned int id, const std::string& db_path)
    {
        {
            auto& file_lock = LockFile::Get(LockFilePath(db_path).c_str());
            std::lock_guard<LockFile> lock(file_lock);
        }

        const auto c = [&db_path]() { return PlainTextDb(db_path); };

        DBMultiThreadedTestWork::WorkItem(id, c, "mp");
    }

    private:
    static std::string LockFilePath(const std::string& db_path) { return db_path + ".test.lock"; }
};

class DbMultiFileTest : public DbTest
{
    protected:
    const std::string user_db_path = temp_file.Path() + ".user";

    void ResetDb() const
    {
        DbTest::ResetDb();
        (void)std::ofstream(user_db_path);
    }
};

template <bool merge_records>
class DbMultiFileReadTest : public DbMultiFileTest
{
    public:
    void Run() const
    {
        std::cout << "Running multifile read test";
        if(merge_records)
            std::cout << " with merge";
        std::cout << "..." << std::endl;

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
    static const std::array<std::pair<const std::string, TestData>, 1>& single_item_data()
    {
        static const std::array<std::pair<const std::string, TestData>, 1> data{
            {{id0(), value2()}}};

        return data;
    }

    void MergedAndMissing() const
    {
        RawWrite(temp_file, key(), common_data());
        RawWrite(user_db_path, key(), single_item_data());

        static const std::array<std::pair<const std::string, TestData>, 2> merged_data{{
            {id1(), value1()}, {id0(), value2()},
        }};

        MultiFileDb<PlainTextDb, PlainTextDb, merge_records> db(temp_file, user_db_path);
        if(merge_records)
            ValidateSingleEntry(key(), merged_data, db);
        else
            ValidateSingleEntry(key(), single_item_data(), db);

        const TestData invalid_key(100, 200);
        auto record1 = db.FindRecord(invalid_key);
        EXPECT(!record1);
    }

    void ReadUser() const
    {
        RawWrite(user_db_path, key(), single_item_data());
        ValidateSingleEntry(
            key(),
            single_item_data(),
            MultiFileDb<PlainTextDb, PlainTextDb, merge_records>(temp_file, user_db_path));
    }

    void ReadInstalled() const
    {
        RawWrite(temp_file, key(), single_item_data());
        ValidateSingleEntry(
            key(),
            single_item_data(),
            MultiFileDb<PlainTextDb, PlainTextDb, merge_records>(temp_file, user_db_path));
    }

    void ReadConflict() const
    {
        RawWrite(temp_file, key(), single_item_data());
        ReadUser();
    }
};

class DbMultiFileWriteTest : public DbMultiFileTest
{
    public:
    void Run() const
    {
        std::cout << "Running multifile write test..." << std::endl;

        ResetDb();

        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            MultiFileDb<PlainTextDb, PlainTextDb, true> db(temp_file, user_db_path);

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(!std::getline(std::ifstream(temp_file), read).good());
        EXPECT(std::getline(std::ifstream(user_db_path), read).good());

        ValidateSingleEntry(key(),
                            common_data(),
                            MultiFileDb<PlainTextDb, PlainTextDb, true>(temp_file, user_db_path));
    }
};

class DbMultiFileOperationsTest : public DbMultiFileTest
{
    public:
    void Run() const
    {
        ResetDb();
        PrepareDb();
        UpdateTest();
        LoadTest();
        RemoveTest();
        RemoveRecordTest();
    }

    void PrepareDb() const
    {
        std::cout << "Running multifile operations test..." << std::endl;

        {
            DbRecord record(key());
            EXPECT(record.SetValues(id0(), value0()));
            EXPECT(record.SetValues(id1(), value2()));

            PlainTextDb db(temp_file);
            EXPECT(db.StoreRecord(record));
        }
    }

    void UpdateTest() const
    {
        std::cout << "Update test..." << std::endl;

        {
            MultiFileDb<PlainTextDb, PlainTextDb, true> db(temp_file, user_db_path);
            EXPECT(db.Update(key(), id1(), value1()));
        }

        {
            PlainTextDb db(user_db_path);
            TestData read(TestData::NoInit{});
            EXPECT(!db.Load(key(), id0(), read));
            EXPECT(db.Load(key(), id1(), read));
            EXPECT_EQUAL(read, value1());
        }

        {
            PlainTextDb db(temp_file);
            ValidateData(db, value2());
        }
    }

    void LoadTest() const
    {
        std::cout << "Load test..." << std::endl;

        MultiFileDb<PlainTextDb, PlainTextDb, true> db(temp_file, user_db_path);
        ValidateData(db, value1());
    }

    void RemoveTest() const
    {
        std::cout << "Remove test..." << std::endl;

        MultiFileDb<PlainTextDb, PlainTextDb, true> db(temp_file, user_db_path);
        EXPECT(!db.Remove(key(), id0()));
        EXPECT(db.Remove(key(), id1()));

        ValidateData(db, value2());
    }

    void RemoveRecordTest() const
    {
        std::cout << "Remove record test..." << std::endl;

        MultiFileDb<PlainTextDb, PlainTextDb, true> db(temp_file, user_db_path);
        EXPECT(db.Update(key(), id1(), value1()));
        EXPECT(db.RemoveRecord(key()));

        ValidateData(db, value2());
    }

    template <class TDb>
    void ValidateData(TDb& db, const TestData& id1Value) const
    {
        TestData read(TestData::NoInit{});
        EXPECT(db.Load(key(), id0(), read));
        EXPECT_EQUAL(read, value0());
        EXPECT(db.Load(key(), id1(), read));
        EXPECT_EQUAL(read, id1Value);
    }
};

class DbMultiFileMultiThreadedReadTest : public DbMultiFileTest
{
    public:
    void Run() const
    {
        std::cout << "Testing db for multifile multithreaded read access..." << std::endl;

        std::mutex mutex;
        std::vector<std::thread> threads;

        std::cout << "Initializing test data..." << std::endl;
        const std::string p = temp_file;
        const auto& up      = user_db_path;
        const auto c = [&p, up]() { return MultiFileDb<PlainTextDb, PlainTextDb, true>(p, up); };
        ResetDb();
        DBMultiThreadedTestWork::FillForReading(c);

        std::cout << "Launching test threads..." << std::endl;
        threads.reserve(DBMultiThreadedTestWork::threads_count);

        {
            std::unique_lock<std::mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
                threads.emplace_back([c, &mutex, i]() {
                    (void)std::unique_lock<std::mutex>(mutex);
                    DBMultiThreadedTestWork::ReadWorkItem(i, c, "mt");
                });
        }

        std::cout << "Waiting for test threads..." << std::endl;
        for(auto& thread : threads)
            thread.join();
    }
};

class DbMultiFileMultiThreadedTest : public DbMultiFileTest
{
    public:
    static constexpr const char* logs_path_arg = "thread-logs-root";

    void Run() const
    {
        std::cout << "Testing db for multifile multithreaded write access..." << std::endl;

        ResetDb();
        std::mutex mutex;
        std::vector<std::thread> threads;

        std::cout << "Initializing test data..." << std::endl;
        DBMultiThreadedTestWork::Initialize();

        std::cout << "Launching test threads..." << std::endl;
        threads.reserve(DBMultiThreadedTestWork::threads_count);
        const std::string p = temp_file;
        const auto up       = user_db_path;
        const auto c = [&p, &up]() { return MultiFileDb<PlainTextDb, PlainTextDb, true>(p, up); };

        {
            std::unique_lock<std::mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
                threads.emplace_back([c, &mutex, i]() {
                    (void)std::unique_lock<std::mutex>(mutex);
                    DBMultiThreadedTestWork::WorkItem(i, c, "mt");
                });
        }

        std::cout << "Waiting for test threads..." << std::endl;
        for(auto& thread : threads)
            thread.join();

        std::cout << "Validating results..." << std::endl;
        DBMultiThreadedTestWork::ValidateCommonPart(c);
        std::cout << "Validation passed..." << std::endl;
    }
};

struct PerfDbDriver : test_driver
{
    PerfDbDriver()
    {
        add(logs_root, DbMultiThreadedTest::logs_path_arg);
        add(test_write, DbMultiProcessTest::write_arg, flag());

        add(mt_child_id, DbMultiProcessTest::id_arg);
        add(mt_child_db_path, DbMultiProcessTest::path_arg);
    }

    void run() const
    {
        if(!logs_root.empty())
            thread_logs_root() = logs_root;

        if(full_set)
        {
            tests::full_set() = true;

#if MIOPEN_BACKEND_HIP
            DBMultiThreadedTestWork::threads_count = 32;
#else
            DBMultiThreadedTestWork::threads_count = 32;
#endif
            DBMultiThreadedTestWork::common_part_size = 128;
            DBMultiThreadedTestWork::unique_part_size = 128;
        }

        if(mt_child_id >= 0)
        {
            DbMultiProcessTest::WorkItem(mt_child_id, mt_child_db_path, test_write);
            return;
        }

        DbFindTest().Run();
        DbStoreTest().Run();
        DbUpdateTest().Run();
        DbRemoveTest().Run();
        DbReadTest().Run();
        DbWriteTest().Run();
        DbOperationsTest().Run();
        DbParallelTest().Run();

        DbMultiThreadedReadTest().Run();
        DbMultiProcessReadTest().Run();
        DbMultiThreadedTest().Run();
        DbMultiProcessTest().Run();
#if !MIOPEN_DISABLE_USERDB
        DbMultiFileReadTest<true>().Run();
        DbMultiFileReadTest<false>().Run();
        DbMultiFileWriteTest().Run();
        DbMultiFileOperationsTest().Run();
        DbMultiFileMultiThreadedReadTest().Run();
        DbMultiFileMultiThreadedTest().Run();
#endif
    }

    private:
    bool test_write = false;
    std::string logs_root;

    int mt_child_id = -1;
    std::string mt_child_db_path;
};

} // namespace tests
} // namespace miopen

int main(int argc, const char* argv[])
{
    miopen::tests::exe_path() = argv[0];
    test_drive<miopen::tests::PerfDbDriver>(argc, argv);
}
