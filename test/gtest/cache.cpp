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

#include <miopen/binary_cache.hpp>
#include <miopen/kern_db.hpp>
#include <miopen/temp_file.hpp>

#include <miopen/md5.hpp>
#include "../test.hpp"
#include "../random.hpp"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

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

// EXPECT() to be repalced by ASSERT_EXIT
// CHECK() to be replaced by EXPECT_TRUE
TEST(TestCache, check_bz2_compress)
{
    std::string to_compress;
    bool success = false;
    std::string cmprsd;

    EXPECT_TRUE(throws([&]() { cmprsd = miopen::compress(to_compress, &success); }));

    to_compress = random_string(4096);
    // if the following throws the test will fail
    cmprsd = miopen::compress(to_compress, nullptr);
    ASSERT_TRUE(!(cmprsd.empty()))
        << "Failed: TestCache at check_bz2_compress" << __FILE__ ": " << __LINE__;
    cmprsd = miopen::compress(to_compress, &success);
    ASSERT_TRUE(success) << "Failed: TestCache at check_bz2_compress" << __FILE__ ": " << __LINE__;
    ASSERT_TRUE(cmprsd.size() < to_compress.size())
        << "Failed: TestCache at check_bz2_compress" << __FILE__ ": " << __LINE__;
}

TEST(TestCache, check_bz2_decompress)
{
    std::string empty_string;

    std::string decompressed_str;

    // CHECK(throws([&]() { decompressed_str = miopen::decompress(empty_string, 0); }));
    // EXPECT_THAT([&]() { decompressed_str = miopen::decompress(empty_string, 0); },
    // ::testing::Throws<std::runtime_error>());
    EXPECT_TRUE(throws([&]() { decompressed_str = miopen::decompress(empty_string, 0); }));

    auto orig_str = random_string(4096);
    bool success  = false;
    std::string compressed_str;
    compressed_str = miopen::compress(orig_str, &success);
    ASSERT_TRUE(success == true) << "Failed: TestCache at check_bz2_decompress" << __FILE__ ": "
                                 << __LINE__ << std::endl;

    decompressed_str = miopen::decompress(compressed_str, orig_str.size());
    ASSERT_TRUE(decompressed_str == orig_str)
        << "Failed: TestCache at check_bz2_decompress" << __FILE__ ": " << __LINE__ << std::endl;

    EXPECT_TRUE(throws([&]() { decompressed_str = miopen::decompress(compressed_str, 10); }));

    ASSERT_TRUE(decompressed_str == miopen::decompress(compressed_str, orig_str.size() + 10))
        << "Failed: TestCache at check_bz2_decompress" << __FILE__ ": " << __LINE__ << std::endl;
}

TEST(TestCache, check_kern_db)
{
    miopen::KernelConfig cfg0;
    cfg0.kernel_name = "kernel1";
    cfg0.kernel_args = random_string(512);
    cfg0.kernel_blob = random_string(8192);

    miopen::KernDb empty_db("", false);
    EXPECT_TRUE(empty_db.RemoveRecordUnsafe(cfg0))
        << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__
        << std::endl; // for empty file, remove should succeed
    EXPECT_FALSE(empty_db.FindRecordUnsafe(cfg0))
        << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__
        << std::endl; // no record in empty database
    EXPECT_FALSE(empty_db.StoreRecordUnsafe(cfg0))
        << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__
        << std::endl; // storing in an empty database should fail

    {
        miopen::TempFile temp_file("tmp-kerndb");
        miopen::KernDb clean_db(std::string(temp_file), false);

        EXPECT_TRUE(clean_db.StoreRecordUnsafe(cfg0))
            << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__ << std::endl;
        auto readout = clean_db.FindRecordUnsafe(cfg0);
        EXPECT_TRUE(readout) << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__;
        EXPECT_TRUE(readout.get() == cfg0.kernel_blob)
            << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__ << std::endl;
        EXPECT_TRUE(clean_db.RemoveRecordUnsafe(cfg0))
            << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__ << std::endl;
        EXPECT_FALSE(clean_db.FindRecordUnsafe(cfg0))
            << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__ << std::endl;
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
        EXPECT_TRUE(err_db.StoreRecordUnsafe(cfg0))
            << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__ << std::endl;
        // In which case decompresion should not be called
        EXPECT_TRUE(err_db.FindRecordUnsafe(cfg0))
            << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__ << std::endl;
        EXPECT_TRUE(err_db.RemoveRecordUnsafe(cfg0))
            << "Failed: TestCache at check_kern_db" << __FILE__ ": " << __LINE__ << std::endl;
    }
}
#endif

TEST(TestCache, check_cache_file)
{
    auto p = miopen::GetCacheFile("gfx", "base", "args", false);
    EXPECT_TRUE(p.filename().string() == "base.o")
        << "Failed: TestCache at check_cache_file" << __FILE__ ": " << __LINE__ << std::endl;
}

TEST(TestCache, check_cache_str)
{
    auto p    = miopen::GetCacheFile("gfx", "base", "args", true);
    auto name = miopen::md5("base");
    EXPECT_TRUE(p.filename().string() == name + ".o")
        << "Failed: TestCache at check_cache_str" << __FILE__ ": " << __LINE__ << std::endl;
}
