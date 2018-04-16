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
    static boost::filesystem::path exe_path;
    return exe_path;
}

static boost::optional<std::string>& thread_logs_root()
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static boost::optional<std::string> path(boost::none);
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

    struct NoInit
    {
    };

    inline TestData(NoInit) : x(0), y(0) {}
    inline TestData(Random& rnd) : x(rnd.Next()), y(rnd.Next()) {}
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
        TestData t(NoInit{});
        std::istringstream ss(s);

        const auto success = DeserializeField(ss, &t.x, sep) && DeserializeField(ss, &t.y, sep);

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
        const auto value = std::strtol(start, &end, 10);

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
    DbTest() : temp_file("miopen.tests.perfdb") {}

    virtual ~DbTest() { std::remove(LockFilePath(temp_file.Path()).c_str()); }

    protected:
    TempFile temp_file;

    static const std::array<std::pair<const char*, TestData>, 2>& common_data()
    {
        static const std::array<std::pair<const char*, TestData>, 2> data{
            {/// \ref
             /// https://stackoverflow.com/questions/22501368/why-wasnt-a-double-curly-braces-syntax-preferred-for-constructors-taking-a-std
             std::make_pair(id1(), value1()),
             std::make_pair(id0(), value0())}};

        return data;
    }

    inline void ResetDb() const { (void)std::ofstream(temp_file); }

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

    template <class TKey, class TValue, size_t count>
    static inline void RawWrite(const std::string& db_path,
                                const TKey& key,
                                const std::array<std::pair<const char*, TValue>, count> values)
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
    static inline void ValidateSingleEntry(
        TKey key, const std::array<std::pair<const char*, TValue>, count> values, TDb db)
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
    inline void Run() const
    {
        std::cout << "Testing db for reading premade file by FindRecord..." << std::endl;

        ResetDb();
        RawWrite(temp_file, key(), common_data());

        Db db(temp_file);
        ValidateSingleEntry(key(), common_data(), db);

        const TestData invalid_key(100, 200);
        auto record1 = db.FindRecord(invalid_key);
        EXPECT(!record1);
    }
};

class DbStoreTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for reading stored data..." << std::endl;

        ResetDb();
        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            Db db(temp_file);

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file), read).good());

        ValidateSingleEntry(key(), common_data(), Db(temp_file));
    }
};

class DbUpdateTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for updating existing records..." << std::endl;

        ResetDb();
        // Store record0 (key=id0:value0)
        DbRecord record0(key());
        EXPECT(record0.SetValues(id0(), value0()));

        {
            Db db(temp_file);

            EXPECT(db.StoreRecord(record0));
        }

        // Update with record1 (key=id1:value1)
        DbRecord record1(key());
        EXPECT(record1.SetValues(id1(), value1()));

        {
            Db db(temp_file);

            EXPECT(db.UpdateRecord(record1));
        }

        // Check record1 (key=id0:value0;id1:value1)
        TestData read0, read1;
        EXPECT(record1.GetValues(id0(), read0));
        EXPECT(record1.GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);

        // Check record that is stored in db (key=id0:value0;id1:value1)
        ValidateSingleEntry(key(), common_data(), Db(temp_file));
    }
};

class DbRemoveTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for removing records..." << std::endl;

        ResetDb();
        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            Db db(temp_file);

            EXPECT(db.StoreRecord(record));
        }

        {
            Db db(temp_file);

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

        ResetDb();
        RawWrite(temp_file, key(), common_data());
        ValidateSingleEntry(key(), common_data(), Db(temp_file));
    }
};

class DbWriteTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for storing unexistent records by update..." << std::endl;

        ResetDb();

        {
            Db db(temp_file);

            EXPECT(db.Update(key(), id0(), value0()));
            EXPECT(db.Update(key(), id1(), value1()));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file), read).good());

        ValidateSingleEntry(key(), common_data(), Db(temp_file));
    }
};

class DbOperationsTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing different db operations db..." << std::endl;

        ResetDb();
        const TestData to_be_rewritten(7, 8);

        {
            Db db(temp_file);

            EXPECT(db.Update(key(), id0(), to_be_rewritten));
            EXPECT(db.Update(key(), id1(), to_be_rewritten));

            // Rewritting existing value with other.
            EXPECT(db.Update(key(), id1(), value1()));

            // Rewritting existing value with same. In fact no DB manipulation should be performed
            // inside of store in such case.
            EXPECT(db.Update(key(), id1(), value1()));
        }

        {
            Db db(temp_file);

            // Rewriting existing value to store it to file.
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            TestData read0, read1, read_missing;
            const auto read_missing_cmp(read_missing);
            Db db(temp_file);

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
            Db db(temp_file);

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

        ResetDb();

        {
            Db db(temp_file);
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            Db db0(temp_file);
            Db db1(temp_file);

            auto r0 = db0.FindRecord(key());
            auto r1 = db1.FindRecord(key());

            EXPECT(r0);
            EXPECT(r1);

            EXPECT(r0->SetValues(id1(), value1()));
            EXPECT(r1->SetValues(id2(), value2()));

            EXPECT(db0.UpdateRecord(*r0));
            EXPECT(db1.UpdateRecord(*r1));
        }

        const std::array<std::pair<const char*, TestData>, 3> data{{
            std::make_pair(id0(), value0()),
            std::make_pair(id1(), value1()),
            std::make_pair(id2(), value2()),
        }};

        ValidateSingleEntry(key(), data, Db(temp_file));
    }
};

class DBMultiThreadedTestWork
{
    public:
    static constexpr unsigned int threads_count    = 64;
    static constexpr unsigned int common_part_size = 64;
    static constexpr unsigned int unique_part_size = 64;
    static constexpr unsigned int ids_per_key      = 16;
    static constexpr unsigned int common_part_seed = 435345;

    static inline const std::array<TestData, common_part_size>& common_part()
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        static const std::array<TestData, common_part_size>& ref = common_part_init();
        return ref;
    }

    static inline void Initialize() { (void)common_part(); }

    template <class TDbConstructor>
    static inline void
    WorkItem(unsigned int id, const TDbConstructor& db_constructor, const std::string& log_postfix)
    {
        RedirectLogs(id, log_postfix, [id, &db_constructor]() {
            CommonPart(db_constructor);
            UniquePart(id, db_constructor);
        });
    }

    template <class TDbConstructor>
    static inline void ReadWorkItem(unsigned int id,
                                    const TDbConstructor& db_constructor,
                                    const std::string& log_postfix)
    {
        RedirectLogs(id, log_postfix, [&db_constructor]() { ReadCommonPart(db_constructor); });
    }

    template <class TDbConstructor>
    static inline void FillForReading(const TDbConstructor& db_constructor)
    {
        auto db = db_constructor();
        CommonPartSection(0u, common_part_size, [&db]() { return db; });
    }

    template <class TDbConstructor>
    static inline void ValidateCommonPart(const TDbConstructor& db_constructor)
    {
        auto db       = db_constructor();
        const auto cp = common_part();

        for(auto i = 0u; i < common_part_size; i++)
        {
            const auto key  = i / ids_per_key;
            const auto id   = i % ids_per_key;
            const auto data = cp[i];
            TestData read(TestData::NoInit{});

            EXPECT(db.Load(std::to_string(key), std::to_string(id), read));
            EXPECT_EQUAL(read, data);
        }
    }

    private:
    template <class TWorker>
    static inline void
    RedirectLogs(unsigned int id, const std::string& log_postfix, const TWorker& worker)
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
    static inline void ReadCommonPart(const TDbConstructor& db_constructor)
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
    static inline void
    ReadCommonPartSection(unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        const auto cp = common_part();

        for(auto i = start; i < end; i++)
        {
            const auto key  = i / ids_per_key;
            const auto id   = i % ids_per_key;
            const auto data = cp[i];
            TestData read(TestData::NoInit{});

            EXPECT(db_getter().Load(std::to_string(key), std::to_string(id), read));
            EXPECT_EQUAL(read, data);
        }
    }

    template <class TDbConstructor>
    static inline void CommonPart(const TDbConstructor& db_constructor)
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
    static inline void
    CommonPartSection(unsigned int start, unsigned int end, const TDbGetter& db_getter)
    {
        const auto cp = common_part();

        for(auto i = start; i < end; i++)
        {
            const auto key  = i / ids_per_key;
            const auto id   = i % ids_per_key;
            const auto data = cp[i];

            db_getter().Update(std::to_string(key), std::to_string(id), data);
        }
    }

    template <class TDbConstructor>
    static inline void UniquePart(unsigned int id, const TDbConstructor& db_constructor)
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
    static inline void
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
    static constexpr const char* logs_path_arg = "thread-logs-root";

    inline void Run() const
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
        const auto c        = [&p]() { return Db(p); };

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

        std::cout << "Validation results..." << std::endl;
        DBMultiThreadedTestWork::ValidateCommonPart(c);
    }
};

class DbMultiThreadedReadTest : public DbTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for multithreaded read access..." << std::endl;

        std::mutex mutex;
        std::vector<std::thread> threads;

        std::cout << "Initializing test data..." << std::endl;
        const std::string p = temp_file;
        const auto c        = [&p]() { return Db(p); };
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

    inline void Run() const
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

        const std::string p = temp_file;
        const auto c        = [&p]() { return Db(p); };
        DBMultiThreadedTestWork::ValidateCommonPart(c);
    }

    static inline void WorkItem(unsigned int id, const std::string& db_path, bool write)
    {
        {
            auto& file_lock = LockFile::Get(LockFilePath(db_path).c_str());
            std::lock_guard<LockFile> lock(file_lock);
        }

        const auto c = [&db_path]() { return Db(db_path); };

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
    inline void Run() const
    {
        std::cout << "Testing db for multiprocess read access..." << std::endl;

        std::vector<FILE*> children(DBMultiThreadedTestWork::threads_count);
        const auto lock_file_path = LockFilePath(temp_file);

        std::cout << "Initializing test data..." << std::endl;
        std::string p = temp_file;
        const auto c  = [&p]() { return Db(p); };
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

    static inline void WorkItem(unsigned int id, const std::string& db_path)
    {
        {
            auto& file_lock = LockFile::Get(LockFilePath(db_path).c_str());
            std::lock_guard<LockFile> lock(file_lock);
        }

        const auto c = [&db_path]() { return Db(db_path); };

        DBMultiThreadedTestWork::WorkItem(id, c, "mp");
    }

    private:
    static std::string LockFilePath(const std::string& db_path) { return db_path + ".test.lock"; }
};

class DbMultiFileTest : public DbTest
{
    protected:
    const std::string user_db_path = temp_file.Path() + ".user";

    inline void ResetDb() const
    {
        DbTest::ResetDb();
        (void)std::ofstream(user_db_path);
    }
};

class DbMultiFileReadTest : public DbMultiFileTest
{
    public:
    inline void Run() const
    {
        std::cout << "Running multifile read test..." << std::endl;

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
    static const std::array<std::pair<const char*, TestData>, 1>& single_item_data()
    {
        static const std::array<std::pair<const char*, TestData>, 1> data{
            {std::make_pair(id0(), value0())}};

        return data;
    }

    inline void MergedAndMissing() const
    {
        RawWrite(temp_file, key(), common_data());

        MultiFileDb db(temp_file, user_db_path);
        ValidateSingleEntry(key(), common_data(), db);

        const TestData invalid_key(100, 200);
        auto record1 = db.FindRecord(invalid_key);
        EXPECT(!record1);
    }

    inline void ReadUser() const
    {
        RawWrite(user_db_path, key(), single_item_data());
        ValidateSingleEntry(key(), single_item_data(), MultiFileDb(temp_file, user_db_path));
    }

    inline void ReadInstalled() const
    {
        RawWrite(temp_file, key(), single_item_data());
        ValidateSingleEntry(key(), single_item_data(), MultiFileDb(temp_file, user_db_path));
    }

    inline void ReadConflict() const
    {
        RawWrite(temp_file, key(), single_item_data());
        ReadUser();
    }
};

class DbMultiFileWriteTest : public DbMultiFileTest
{
    public:
    inline void Run() const
    {
        std::cout << "Running multifile write test..." << std::endl;

        ResetDb();

        DbRecord record(key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            MultiFileDb db(temp_file, user_db_path);

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(!std::getline(std::ifstream(temp_file), read).good());
        EXPECT(std::getline(std::ifstream(user_db_path), read).good());

        ValidateSingleEntry(key(), common_data(), MultiFileDb(temp_file, user_db_path));
    }
};

class DbMultiFileMultiThreadedReadTest : public DbMultiFileTest
{
    public:
    inline void Run() const
    {
        std::cout << "Testing db for multifile multithreaded read access..." << std::endl;

        std::mutex mutex;
        std::vector<std::thread> threads;

        std::cout << "Initializing test data..." << std::endl;
        const std::string p = temp_file;
        const auto& up      = user_db_path;
        const auto c        = [&p, up]() { return MultiFileDb(p, up); };
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

    inline void Run() const
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
        const auto c        = [&p, &up]() { return MultiFileDb(p, up); };

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

        std::cout << "Validation results..." << std::endl;
        DBMultiThreadedTestWork::ValidateCommonPart(c);
    }
};

struct PerfDbDriver : test_driver
{
    public:
    PerfDbDriver()
    {
        add(logs_root, miopen::tests::DbMultiThreadedTest::logs_path_arg);
        add(test_write, miopen::tests::DbMultiProcessTest::write_arg, flag());

        add(mt_child_id, DbMultiProcessTest::id_arg);
        add(mt_child_db_path, DbMultiProcessTest::path_arg);
    }

    void run() const
    {
        if(!logs_root.empty())
            thread_logs_root() = logs_root;

        if(mt_child_id >= 0)
        {
            miopen::tests::DbMultiProcessTest::WorkItem(mt_child_id, mt_child_db_path, test_write);
            return;
        }

        miopen::tests::DbFindTest().Run();
        miopen::tests::DbStoreTest().Run();
        miopen::tests::DbUpdateTest().Run();
        miopen::tests::DbRemoveTest().Run();
        miopen::tests::DbReadTest().Run();
        miopen::tests::DbWriteTest().Run();
        miopen::tests::DbOperationsTest().Run();
        miopen::tests::DbParallelTest().Run();
        miopen::tests::DbMultiThreadedReadTest().Run();
        miopen::tests::DbMultiProcessReadTest().Run();
        miopen::tests::DbMultiThreadedTest().Run();
        miopen::tests::DbMultiProcessTest().Run();
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
