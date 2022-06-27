#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>

//Todo: Reduced duplicated code from cache.cpp
//E.g. declare extern check_cache() in cache. cpp and declare dependency on cache.cpp?
#include <miopen/binary_cache.hpp>
#include <miopen/kern_db.hpp>
#include <miopen/temp_file.hpp>

#include <miopen/md5.hpp>
#include "../test.hpp"
#include "../random.hpp"

// Demonstrate some basic assertions.
//Todo: Include a simple task
//TEST_F()?: use the same data configuration for multiple tests

#ifdef __cplusplus
extern "C" {
#endif

//extern void check_cache(void);

#ifdef __cplusplus
}
#endif

#if MIOPEN_ENABLE_SQLITE
std::string random_string(size_t length)
{
    auto randchar = []() -> char {
        const char charset[] = "0123456789"
                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                               "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[GET_RAND() % max_index];
    };
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
}

void check_bz2_compress()
{
    std::string to_compress;
    bool success = false;
    std::string cmprsd;
    CHECK(throws([&]() { cmprsd = miopen::compress(to_compress, &success); }));

    to_compress = random_string(4096);
    // if the following throws the test will fail
    cmprsd = miopen::compress(to_compress, nullptr);
    EXPECT(!(cmprsd.empty()));
    cmprsd = miopen::compress(to_compress, &success);
    EXPECT(success);
    EXPECT(cmprsd.size() < to_compress.size());
}

void check_bz2_decompress()
{
    std::string empty_string;

    std::string decompressed_str;
    CHECK(throws([&]() { decompressed_str = miopen::decompress(empty_string, 0); }));

    auto orig_str = random_string(4096);
    bool success  = false;
    std::string compressed_str;
    compressed_str = miopen::compress(orig_str, &success);
    EXPECT(success == true);

    decompressed_str = miopen::decompress(compressed_str, orig_str.size());
    EXPECT(decompressed_str == orig_str);

    CHECK(throws([&]() { decompressed_str = miopen::decompress(compressed_str, 10); }));

    EXPECT(decompressed_str == miopen::decompress(compressed_str, orig_str.size() + 10));
}

void check_kern_db()
{
    miopen::KernelConfig cfg0;
    cfg0.kernel_name = "kernel1";
    cfg0.kernel_args = random_string(512);
    cfg0.kernel_blob = random_string(8192);

    miopen::KernDb empty_db("", false);
    CHECK(empty_db.RemoveRecordUnsafe(cfg0)); // for empty file, remove should succeed
    CHECK(!empty_db.FindRecordUnsafe(cfg0));  // no record in empty database
    CHECK(!empty_db.StoreRecordUnsafe(cfg0)); // storing in an empty database should fail

    {
        miopen::TempFile temp_file("tmp-kerndb");
        miopen::KernDb clean_db(std::string(temp_file), false);

        CHECK(clean_db.StoreRecordUnsafe(cfg0));
        auto readout = clean_db.FindRecordUnsafe(cfg0);
        CHECK(readout);
        CHECK(readout.get() == cfg0.kernel_blob);
        CHECK(clean_db.RemoveRecordUnsafe(cfg0));
        CHECK(!clean_db.FindRecordUnsafe(cfg0));
    }

    {
        miopen::TempFile temp_file("tmp-kerndb");
        miopen::KernDb err_db(
            std::string(temp_file),
            false,
            [](std::string str, bool* success) {
                std::ignore = str;
                *success    = false;
                return "";
            },
            [](std::string str, unsigned int sz) -> std::string {
                std::ignore = str;
                std::ignore = sz;
                throw;
            }); // error compressing
        // Even if compression fails, it should still work
        CHECK(err_db.StoreRecordUnsafe(cfg0));
        // In which case decompresion should not be called
        CHECK(err_db.FindRecordUnsafe(cfg0));
        CHECK(err_db.RemoveRecordUnsafe(cfg0));
    }
}
#endif

void check_cache_file()
{
    auto p = miopen::GetCacheFile("gfx", "base", "args", false);
    CHECK(p.filename().string() == "base.o");
}

void check_cache_str()
{
    auto p    = miopen::GetCacheFile("gfx", "base", "args", true);
    auto name = miopen::md5("base");
    CHECK(p.filename().string() == name + ".o");
}

TEST(HelloTest, BasicAssertions)
{

    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);

    // Check if we can access MIOpen internals
    auto idx = 0;
    for(const auto& solver_id :
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        std::ignore = solver_id;
        ++idx;
    }
    EXPECT_GT(idx, 0);

    //Test Case for cache.cpp
    idx = 0;
    check_cache_file();
    ++idx;
    check_cache_str();
    ++idx;
#if MIOPEN_ENABLE_SQLITE
    check_bz2_compress();
    ++idx;
    check_bz2_decompress();
    ++idx;
    check_kern_db();
    ++idx;
#endif
    EXPECT_GT(idx, 0);
}

