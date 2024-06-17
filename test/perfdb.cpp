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

#include <miopen/filesystem.hpp>
#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/lock_file.hpp>
#include <miopen/process.hpp>
#include <miopen/ramdb.hpp>
#include <miopen/readonlyramdb.hpp>
#include <miopen/temp_file.hpp>

#include <boost/optional.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <limits>
#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

namespace miopen {
namespace tests {

struct TestRordbEmbedFsOverrideLock
{
    TestRordbEmbedFsOverrideLock() : cached(debug::rordb_embed_fs_override())
    {
        debug::rordb_embed_fs_override() = true;
    }

    ~TestRordbEmbedFsOverrideLock() { debug::rordb_embed_fs_override() = cached; }

private:
    bool cached;
};

static fs::path& exe_path()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static fs::path exe_path;
    return exe_path;
}

static boost::optional<fs::path>& thread_logs_root()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static boost::optional<fs::path> path(boost::none);
    return path;
}

static bool& full_set()
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static bool full_set = false;
    return full_set;
}

struct ArgsHelper
{
    static constexpr const char* logs_path_arg = "thread-logs-root";
    static constexpr const char* write_arg     = "mp-test-child-write";
    static constexpr const char* id_arg        = "mp-test-child";
    static constexpr const char* path_arg      = "mp-test-child-path";
    static constexpr const char* db_class_arg  = "mp-test-child-db-path";

    struct db_class
    {
        static constexpr const char* db    = "db";
        static constexpr const char* ramdb = "ramdb";

        template <class TDb>
        static constexpr std::enable_if_t<std::is_same<TDb, PlainTextDb>::value, const char*> Get()
        {
            return db;
        }

        template <class TDb>
        static constexpr std::enable_if_t<std::is_same<TDb, RamDb>::value, const char*> Get()
        {
            return ramdb;
        }
    };
};

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

    template <class TSelf, class Visitor>
    static void VisitAll(TSelf&& self, Visitor visitor)
    {
        visitor(self.x, "x");
        visitor(self.y, "y");
    }

    void Serialize(std::ostream& s) const
    {
        static const auto sep = 'x';
        s << x << sep << y;
    }

    bool Deserialize(const std::string& s)
    {
        static const auto sep = 'x';
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
    DbTest(TempFile& temp_file_) : temp_file(temp_file_) { ResetDb(); }

    virtual ~DbTest() { fs::remove(LockFilePath(temp_file.Path())); }

protected:
    TempFile& temp_file;

    static const std::array<std::pair<const std::string, TestData>, 2>& common_data()
    {
        static const std::array<std::pair<const std::string, TestData>, 2> data{{
            {id1(), value1()},
            {id0(), value0()},
        }};

        return data;
    }

    static void ResetDbFile(TempFile& tmp_file) { tmp_file = TempFile{tmp_file.GetPathInfix()}; }

    void ResetDb() { ResetDbFile(temp_file); }

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
    static void RawWrite(const fs::path& db_path,
                         const TKey& key,
                         const std::array<std::pair<const std::string, TValue>, count> values)
    {
        std::ostringstream ss_vals;
        ss_vals << key.x << 'x' << key.y << '=';

        auto first = true;

        for(const auto& id_value : values)
        {
            if(!first)
                ss_vals << ";";

            first = false;
            ss_vals << id_value.first << ':' << id_value.second.x << 'x' << id_value.second.y;
        }

        std::ofstream(db_path, std::ios::out | std::ios::ate) << ss_vals.str() << std::endl;
    }

    template <class TDb, class TKey, class TValue, size_t count>
    static void ValidateSingleEntry(
        TKey key, const std::array<std::pair<const std::string, TValue>, count> values, TDb& db)
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

template <class TDb>
class DbFindTest : public DbTest
{
public:
    DbFindTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for reading premade file by FindRecord...");

        RawWrite(temp_file, key(), common_data());

        TDb db(DbKinds::PerfDb, temp_file);
        ValidateSingleEntry(key(), common_data(), db);

        const TestData invalid_key(100, 200);
        auto record1 = db.FindRecord(invalid_key);
        EXPECT(!record1);
    }
};

template <class TDb>
class DbStoreTest : public DbTest
{
public:
    DbStoreTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for reading stored data...");

        DbRecord record(DbKinds::PerfDb, key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            TDb db(DbKinds::PerfDb, temp_file);

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file.Path()), read).good());

        TDb db{DbKinds::PerfDb, temp_file};
        ValidateSingleEntry(key(), common_data(), db);
    }
};

template <class TDb>
class DbUpdateTest : public DbTest
{
public:
    DbUpdateTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for updating existing records...");

        // Store record0 (key=id0:value0)
        DbRecord record0(DbKinds::PerfDb, key());
        EXPECT(record0.SetValues(id0(), value0()));

        {
            TDb db(DbKinds::PerfDb, temp_file);

            EXPECT(db.StoreRecord(record0));
        }

        // Update with record1 (key=id1:value1)
        DbRecord record1(DbKinds::PerfDb, key());
        EXPECT(record1.SetValues(id1(), value1()));

        {
            TDb db(DbKinds::PerfDb, temp_file);

            EXPECT(db.UpdateRecord(record1));
        }

        // Check record1 (key=id0:value0;id1:value1)
        TestData read0, read1;
        EXPECT(record1.GetValues(id0(), read0));
        EXPECT(record1.GetValues(id1(), read1));
        EXPECT_EQUAL(value0(), read0);
        EXPECT_EQUAL(value1(), read1);

        // Check record that is stored in db (key=id0:value0;id1:value1)
        TDb db{DbKinds::PerfDb, temp_file};
        ValidateSingleEntry(key(), common_data(), db);
    }
};

template <class TDb>
class DbRemoveTest : public DbTest
{
public:
    DbRemoveTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for removing records...");

        DbRecord record(DbKinds::PerfDb, key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            TDb db(DbKinds::PerfDb, temp_file);

            EXPECT(db.StoreRecord(record));
        }

        {
            TDb db(DbKinds::PerfDb, temp_file);

            EXPECT(db.FindRecord(key()));
            EXPECT(db.RemoveRecord(key()));
            EXPECT(!db.FindRecord(key()));
        }
    }
};

template <class TDb>
class DbReadTest : public DbTest
{
public:
    DbReadTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for reading premade file by Load...");

        RawWrite(temp_file, key(), common_data());
        TDb db{DbKinds::PerfDb, temp_file};
        ValidateSingleEntry(key(), common_data(), db);
    }
};

template <class TDb>
class DbWriteTest : public DbTest
{
public:
    DbWriteTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for storing unexistent records by update...");

        {
            TDb db(DbKinds::PerfDb, temp_file);

            EXPECT(db.Update(key(), id0(), value0()));
            EXPECT(db.Update(key(), id1(), value1()));
        }

        std::string read;
        EXPECT(std::getline(std::ifstream(temp_file.Path()), read).good());

        TDb db{DbKinds::PerfDb, temp_file};
        ValidateSingleEntry(key(), common_data(), db);
    }
};

template <class TDb>
class DbOperationsTest : public DbTest
{
public:
    DbOperationsTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing different " << ArgsHelper::db_class::Get<TDb>()
                                               << " operations...");

        const TestData to_be_rewritten(7, 8);

        {
            TDb db(DbKinds::PerfDb, temp_file);

            EXPECT(db.Update(key(), id0(), to_be_rewritten));
            EXPECT(db.Update(key(), id1(), to_be_rewritten));

            // Rewritting existing value with other.
            EXPECT(db.Update(key(), id1(), value1()));

            // Rewritting existing value with same. In fact no DB manipulation should be performed
            // inside of store in such case.
            EXPECT(db.Update(key(), id1(), value1()));
        }

        {
            TDb db(DbKinds::PerfDb, temp_file);

            // Rewriting existing value to store it to file.
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            TestData read0, read1, read_missing;
            const auto read_missing_cmp(read_missing);
            TDb db(DbKinds::PerfDb, temp_file);

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
            TDb db(DbKinds::PerfDb, temp_file);

            EXPECT(!db.Load(key(), id0(), read0));
            EXPECT(db.Load(key(), id1(), read1));

            EXPECT_EQUAL(read0, read_missing_cmp);
            EXPECT_EQUAL(read1, value1());
        }
    }
};

template <class TDb>
class DbParallelTest : public DbTest
{
public:
    DbParallelTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default,
            "Test",
            "Testing " << ArgsHelper::db_class::Get<TDb>()
                       << " for using two objects targeting one file existing in one scope...");

        {
            TDb db(DbKinds::PerfDb, temp_file);
            EXPECT(db.Update(key(), id0(), value0()));
        }

        {
            TDb db0(DbKinds::PerfDb, temp_file);
            TDb db1(DbKinds::PerfDb, temp_file);

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
            {id0(), value0()},
            {id1(), value1()},
            {id2(), value2()},
        }};

        TDb db{DbKinds::PerfDb, temp_file};
        ValidateSingleEntry(key(), data, db);
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
        decltype(db_constructor()) db = db_constructor();
        CommonPartSection(0u, common_part_size, [&db]() -> decltype(db) { return db; });
    }

    template <class TDbConstructor>
    static void ValidateCommonPart(const TDbConstructor& db_constructor)
    {
        decltype(db_constructor()) db = db_constructor();
        const auto cp                 = common_part();

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
                *thread_logs_root() / ("thread-" + std::to_string(id) + "_" + log_postfix + ".log");
            const auto err_path = *thread_logs_root() /
                                  ("thread-" + std::to_string(id) + "_" + log_postfix + "-err.log");

            fs::remove(out_path);
            fs::remove(err_path);

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
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Common part. Section with common db instance.");
        {
            decltype(db_constructor()) db = db_constructor();
            ReadCommonPartSection(0u, common_part_size / 2, [&db]() -> decltype(db) { return db; });
        }

        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Common part. Section with separate db instances.");
        ReadCommonPartSection(
            common_part_size / 2,
            common_part_size,
            [&db_constructor]() -> decltype(db_constructor()) { return db_constructor(); });
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
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Common part. Section with common db instance.");
        {
            decltype(db_constructor()) db = db_constructor();
            CommonPartSection(0u, common_part_size / 2, [&db]() -> decltype(db) { return db; });
        }

        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Common part. Section with separate db instances.");
        CommonPartSection(
            common_part_size / 2,
            common_part_size,
            [&db_constructor]() -> decltype(db_constructor()) { return db_constructor(); });
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

        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Unique part. Section with common db instance.");
        {
            decltype(db_constructor()) db = db_constructor();
            UniquePartSection(rnd, 0, unique_part_size / 2, [&db]() -> decltype(db) { return db; });
        }

        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Unique part. Section with separate db instances.");
        UniquePartSection(
            rnd,
            unique_part_size / 2,
            unique_part_size,
            [&db_constructor]() -> decltype(db_constructor()) { return db_constructor(); });
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

        for(auto i = 0u; i < common_part_size; i++)
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

template <class TDb>
class DbMultiThreadedTest : public DbTest
{
public:
    DbMultiThreadedTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for multithreaded write access...");

        std::shared_mutex mutex;
        std::vector<std::thread> threads;

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        DBMultiThreadedTestWork::Initialize();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test threads...");
        threads.reserve(DBMultiThreadedTestWork::threads_count);
        const auto c = [this]()
            MIOPEN_RETURNS(GetDbInstance<TDb>(DbKinds::PerfDb, temp_file, false));

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
            {
                auto thread_body = [c, &mutex, i]() {
                    std::shared_lock<std::shared_mutex> lock(mutex);
                    DBMultiThreadedTestWork::WorkItem(i, c, "mt");
                };

                threads.emplace_back(thread_body);
            }
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test threads...");
        for(auto& thread : threads)
            thread.join();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validating results...");
        DBMultiThreadedTestWork::ValidateCommonPart(c);
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validation passed...");
    }
};

template <class TDb>
class DbMultiThreadedReadTest : public DbTest
{
public:
    DbMultiThreadedReadTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for multithreaded read access...");

        std::shared_mutex mutex;
        std::vector<std::thread> threads;

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        const auto c = [this]()
            MIOPEN_RETURNS(GetDbInstance<TDb>(DbKinds::PerfDb, temp_file, false));
        DBMultiThreadedTestWork::FillForReading(c);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test threads...");
        threads.reserve(DBMultiThreadedTestWork::threads_count);

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
            {
                threads.emplace_back([c, &mutex, i]() {
                    std::shared_lock<std::shared_mutex> lock(mutex);
                    DBMultiThreadedTestWork::ReadWorkItem(i, c, "mt");
                });
            }
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test threads...");
        for(auto& thread : threads)
            thread.join();
    }
};

template <class TDb>
class DbMultiProcessTest : public DbTest
{
public:
    DbMultiProcessTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for multiprocess write access...");

        std::vector<Process> children{};
        const auto lock_file_path = LockFilePath(temp_file);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        DBMultiThreadedTestWork::Initialize();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test processes...");
        {
            auto& file_lock = LockFile::Get(lock_file_path);
            std::shared_lock<LockFile> lock(file_lock);

            auto id = 0;

            // clang-format off
            for(auto i = 0; i < DBMultiThreadedTestWork::threads_count; ++i)
            {
                auto args =
                    std::string{"--"} + ArgsHelper::write_arg +
                                " --" + ArgsHelper::id_arg + " " + std::to_string(id++) +
                                " --" + ArgsHelper::path_arg + " " + temp_file.Path() +
                                " --" + ArgsHelper::db_class_arg + " " + ArgsHelper::db_class::Get<TDb>();

                if(thread_logs_root())
                {
                    args += std::string{" --"} + ArgsHelper::logs_path_arg + " " + *thread_logs_root();
                }

                if(full_set())
                    args += " --all";

                std::ignore = children.emplace_back(exe_path()).Arguments(args).Execute();
            }
            // clang-format on
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test processes...");
        for(auto&& child : children)
        {
            EXPECT_EQUAL(child.Wait(), 0);
        }

        fs::remove(lock_file_path);

        const auto c = [this]()
            MIOPEN_RETURNS(GetDbInstance<TDb>(DbKinds::PerfDb, temp_file, false));

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validating results...");
        DBMultiThreadedTestWork::ValidateCommonPart(c);
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validation passed...");
    }

    static void WorkItem(unsigned int id, const fs::path& db_path, bool write)
    {
        {
            auto& file_lock = LockFile::Get(LockFilePath(db_path));
            std::lock_guard<LockFile> lock(file_lock);
        }

        const auto c = [&db_path]()
            MIOPEN_RETURNS(GetDbInstance<TDb>(DbKinds::PerfDb, db_path, false));

        if(write)
            DBMultiThreadedTestWork::WorkItem(id, c, "mp");
        else
            DBMultiThreadedTestWork::ReadWorkItem(id, c, "mp");
    }

private:
    static fs::path LockFilePath(const fs::path& db_path) { return db_path + ".test.lock"; }
};

template <class TDb>
class DbMultiProcessReadTest : public DbTest
{
public:
    DbMultiProcessReadTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing " << ArgsHelper::db_class::Get<TDb>()
                                     << " for multiprocess read access...");

        std::vector<Process> children{};
        const auto lock_file_path = LockFilePath(temp_file);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        const auto c = [this]()
            MIOPEN_RETURNS(GetDbInstance<TDb>(DbKinds::PerfDb, temp_file, false));
        DBMultiThreadedTestWork::FillForReading(c);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test processes...");
        {
            auto& file_lock = LockFile::Get(lock_file_path);
            std::shared_lock<LockFile> lock(file_lock);

            auto id = 0;

            // clang-format off
            for(auto i = 0; i < DBMultiThreadedTestWork::threads_count; ++i)
            {
                auto args =
                    std::string{"--"} + ArgsHelper::id_arg + " " + std::to_string(id++) +
                               " --" + ArgsHelper::path_arg + " " + temp_file +
                               " --" + ArgsHelper::db_class_arg + " " + ArgsHelper::db_class::Get<TDb>();

                if(thread_logs_root())
                {
                    args += std::string{" --"} + ArgsHelper::logs_path_arg + " " + *thread_logs_root();
                }

                if(full_set())
                    args += " --all";

                MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", exe_path() + " " + args);
                std::ignore = children.emplace_back(exe_path()).Arguments(args).Execute();
            }
            // clang-format on
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test processes...");
        for(auto&& child : children)
        {
            EXPECT_EQUAL(child.Wait(), 0);
        }

        fs::remove(lock_file_path);
    }

    static void WorkItem(unsigned int id, const fs::path& db_path)
    {
        {
            auto& file_lock = LockFile::Get(LockFilePath(db_path));
            std::lock_guard<LockFile> lock(file_lock);
        }

        const auto c = [&db_path]() { return TDb(db_path, false); };

        DBMultiThreadedTestWork::WorkItem(id, c, "mp");
    }

private:
    static fs::path LockFilePath(const fs::path& db_path) { return db_path + ".test.lock"; }
};

class DbMultiFileTest : public DbTest
{
protected:
    DbMultiFileTest(TempFile& temp_file_) : DbTest(temp_file_) {}

    fs::path user_db_path = temp_file.Path() + ".user";

    void ResetDb()
    {
        DbTest::ResetDb();
        user_db_path = temp_file.Path() + ".user";
    }

private:
#if MIOPEN_EMBED_DB
    TestRordbEmbedFsOverrideLock rordb_embed_fs_override;
#endif
};

template <bool merge_records>
class DbMultiFileReadTest : public DbMultiFileTest
{
public:
    DbMultiFileReadTest(TempFile& temp_file_) : DbMultiFileTest(temp_file_) {}

    void Run()
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Running multifile read test" << (merge_records ? " with merge" : "")
                                                        << "...");

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
            {id1(), value1()},
            {id0(), value2()},
        }};

        MultiFileDb<ReadonlyRamDb, RamDb, merge_records> db(
            DbKinds::PerfDb, temp_file, user_db_path);
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
        MultiFileDb<ReadonlyRamDb, RamDb, merge_records> db(
            DbKinds::PerfDb, temp_file, user_db_path);
        ValidateSingleEntry(key(), single_item_data(), db);
    }

    void ReadInstalled() const
    {
        RawWrite(temp_file, key(), single_item_data());
        MultiFileDb<ReadonlyRamDb, RamDb, merge_records> db(
            DbKinds::PerfDb, temp_file, user_db_path);
        ValidateSingleEntry(key(), single_item_data(), db);
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
    DbMultiFileWriteTest(TempFile& temp_file_) : DbMultiFileTest(temp_file_) {}

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Running multifile write test...");

        DbRecord record(DbKinds::PerfDb, key());
        EXPECT(record.SetValues(id0(), value0()));
        EXPECT(record.SetValues(id1(), value1()));

        {
            MultiFileDb<ReadonlyRamDb, RamDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);

            EXPECT(db.StoreRecord(record));
        }

        std::string read;
        EXPECT(!std::getline(std::ifstream(temp_file.Path()), read).good());
        EXPECT(std::getline(std::ifstream(user_db_path), read).good());

        auto db = MultiFileDb<ReadonlyRamDb, RamDb, true>{DbKinds::PerfDb, temp_file, user_db_path};
        ValidateSingleEntry(key(), common_data(), db);
    }
};

class DbMultiFileOperationsTest : public DbMultiFileTest
{
public:
    DbMultiFileOperationsTest(TempFile& temp_file_) : DbMultiFileTest(temp_file_) {}

    void Run() const
    {
        PrepareDb();
        UpdateTest();
        LoadTest();
        RemoveTest();
        RemoveRecordTest();
    }

    void PrepareDb() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Running multifile operations test...");

        {
            DbRecord record(DbKinds::PerfDb, key());
            EXPECT(record.SetValues(id0(), value0()));
            EXPECT(record.SetValues(id1(), value2()));

            PlainTextDb db(DbKinds::PerfDb, temp_file);
            EXPECT(db.StoreRecord(record));
        }
    }

    void UpdateTest() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Update test...");

        {
            MultiFileDb<ReadonlyRamDb, RamDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
            EXPECT(db.Update(key(), id1(), value1()));
        }

        {
            PlainTextDb db(DbKinds::PerfDb, user_db_path);
            TestData read(TestData::NoInit{});
            EXPECT(!db.Load(key(), id0(), read));
            EXPECT(db.Load(key(), id1(), read));
            EXPECT_EQUAL(read, value1());
        }

        {
            PlainTextDb db(DbKinds::PerfDb, temp_file);
            ValidateData(db, value2());
        }
    }

    void LoadTest() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Load test...");

        MultiFileDb<ReadonlyRamDb, RamDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
        ValidateData(db, value1());
    }

    void RemoveTest() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Remove test...");

        MultiFileDb<ReadonlyRamDb, RamDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
        EXPECT(!db.Remove(key(), id0()));
        EXPECT(db.Remove(key(), id1()));

        ValidateData(db, value2());
    }

    void RemoveRecordTest() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Remove record test...");

        MultiFileDb<ReadonlyRamDb, RamDb, true> db(DbKinds::PerfDb, temp_file, user_db_path);
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
    DbMultiFileMultiThreadedReadTest(TempFile& temp_file_) : DbMultiFileTest(temp_file_) {}

    void Run()
    {
        MIOPEN_LOG_CUSTOM(
            LoggingLevel::Default, "Test", "Testing db for multifile multithreaded read access...");

        std::shared_mutex mutex;
        std::vector<std::thread> threads;

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        const auto c = [this]() {
            return MultiFileDb<ReadonlyRamDb, RamDb, true>(
                DbKinds::PerfDb, temp_file, user_db_path);
        };
        ResetDb();
        DBMultiThreadedTestWork::FillForReading(c);

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test threads...");
        threads.reserve(DBMultiThreadedTestWork::threads_count);

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
            {
                threads.emplace_back([c, &mutex, i]() {
                    std::shared_lock<std::shared_mutex> lock(mutex);
                    DBMultiThreadedTestWork::ReadWorkItem(i, c, "mt");
                });
            }
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test threads...");
        for(auto& thread : threads)
            thread.join();
    }
};

class DbMultiFileMultiThreadedTest : public DbMultiFileTest
{
public:
    DbMultiFileMultiThreadedTest(TempFile& temp_file_) : DbMultiFileTest(temp_file_) {}

    static constexpr const char* logs_path_arg = "thread-logs-root";

    void Run() const
    {
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default,
                          "Test",
                          "Testing db for multifile multithreaded write access...");

        std::shared_mutex mutex;
        std::vector<std::thread> threads;

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Initializing test data...");
        DBMultiThreadedTestWork::Initialize();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Launching test threads...");
        threads.reserve(DBMultiThreadedTestWork::threads_count);
        const auto c = [this]() {
            return MultiFileDb<ReadonlyRamDb, RamDb, true>(
                DbKinds::PerfDb, temp_file, user_db_path);
        };

        {
            std::unique_lock<std::shared_mutex> lock(mutex);

            for(auto i = 0u; i < DBMultiThreadedTestWork::threads_count; i++)
            {
                threads.emplace_back([c, &mutex, i]() {
                    std::shared_lock<std::shared_mutex> lock(mutex);
                    DBMultiThreadedTestWork::WorkItem(i, c, "mt");
                });
            }
        }

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Waiting for test threads...");
        for(auto& thread : threads)
            thread.join();

        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validating results...");
        DBMultiThreadedTestWork::ValidateCommonPart(c);
        MIOPEN_LOG_CUSTOM(LoggingLevel::Default, "Test", "Validation passed...");
    }
};

struct PerfDbDriver : test_driver
{
    PerfDbDriver()
    {
        add(logs_root, ArgsHelper::logs_path_arg);
        add(test_write, ArgsHelper::write_arg, flag());

        add(mt_child_id, ArgsHelper::id_arg);
        add(mt_child_db_path, ArgsHelper::path_arg);
        add(mt_child_db_class, ArgsHelper::db_class_arg);
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
            if(mt_child_db_class == ArgsHelper::db_class::db)
            {
                DbMultiProcessTest<PlainTextDb>::WorkItem(
                    mt_child_id, mt_child_db_path, test_write);
            }
            else if(mt_child_db_class == ArgsHelper::db_class::ramdb)
            {
                DbMultiProcessTest<RamDb>::WorkItem(mt_child_id, mt_child_db_path, test_write);
            }
            return;
        }

        TempFile temp_file{"miopen.tests.perfdb"};

        DbTests<RamDb>(temp_file);
        DbTests<PlainTextDb>(temp_file);
        MultiFileDbTests(temp_file);
    }

private:
    bool test_write = false;
    std::string logs_root;

    int mt_child_id = -1;
    fs::path mt_child_db_path;
    std::string mt_child_db_class;

    template <class TDb>
    void DbTests(TempFile& temp_file) const
    {
        DbFindTest<TDb>{temp_file}.Run();
        DbStoreTest<TDb>{temp_file}.Run();
        DbUpdateTest<TDb>{temp_file}.Run();
        DbRemoveTest<TDb>{temp_file}.Run();
        DbReadTest<TDb>{temp_file}.Run();
        DbWriteTest<TDb>{temp_file}.Run();
        DbOperationsTest<TDb>{temp_file}.Run();
        DbParallelTest<TDb>{temp_file}.Run();

        DbMultiThreadedReadTest<TDb>{temp_file}.Run();
        DbMultiProcessReadTest<TDb>{temp_file}.Run();
        DbMultiThreadedTest<TDb>{temp_file}.Run();
        DbMultiProcessTest<TDb>{temp_file}.Run();
    }

    void MultiFileDbTests(TempFile& temp_file) const
    {
        if(!DisableUserDbFileIO)
        {
            DbMultiFileReadTest<true>{temp_file}.Run();
            DbMultiFileReadTest<false>{temp_file}.Run();
            DbMultiFileWriteTest{temp_file}.Run();
        }
        DbMultiFileOperationsTest{temp_file}.Run();
        DbMultiFileMultiThreadedReadTest{temp_file}.Run();
        DbMultiFileMultiThreadedTest{temp_file}.Run();
    }
};

} // namespace tests
} // namespace miopen

int main(int argc, const char* argv[])
{
    miopen::tests::exe_path() = argv[0];
    test_drive<miopen::tests::PerfDbDriver>(argc, argv);
}
