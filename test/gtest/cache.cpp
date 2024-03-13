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

#include "test.hpp"
#include "random.hpp"

#include <gtest/gtest.h>

#if MIOPEN_ENABLE_SQLITE
std::string random_string(size_t length)
{
    auto randchar = []() -> char {
        const char charset[] = "0123456789"
                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                               "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[prng::gen_0_to_B(max_index)];
    };
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
}

TEST(TestCache, check_bz2_compress)
{
    std::string to_compress;
    bool success = false;
    std::string cmprsd;

    EXPECT_TRUE(throws([&]() { cmprsd = miopen::compress(to_compress, &success); }));

    to_compress = random_string(4096);
    // if the following throws the test will fail
    cmprsd = miopen::compress(to_compress, nullptr);
    ASSERT_TRUE(!(cmprsd.empty()));
    cmprsd = miopen::compress(to_compress, &success);
    ASSERT_TRUE(success);
    ASSERT_TRUE(cmprsd.size() < to_compress.size());
}

TEST(TestCache, check_bz2_decompress)
{
    std::string empty_string;

    std::string decompressed_str;

    EXPECT_TRUE(throws([&]() { decompressed_str = miopen::decompress(empty_string, 0); }));

    auto orig_str = random_string(4096);
    bool success  = false;
    std::string compressed_str;
    compressed_str = miopen::compress(orig_str, &success);
    ASSERT_TRUE(success);

    decompressed_str = miopen::decompress(compressed_str, orig_str.size());
    ASSERT_TRUE(decompressed_str == orig_str);

    EXPECT_TRUE(throws([&]() { decompressed_str = miopen::decompress(compressed_str, 10); }));

    ASSERT_TRUE(decompressed_str == miopen::decompress(compressed_str, orig_str.size() + 10));
}

TEST(TestCache, check_kern_db)
{
    miopen::KernelConfig cfg0;
    cfg0.kernel_name = "kernel1";
    cfg0.kernel_args = random_string(512);
    cfg0.kernel_blob = random_string(8192);

    miopen::KernDb empty_db(miopen::DbKinds::KernelDb, "", false);
    EXPECT_TRUE(empty_db.RemoveRecordUnsafe(cfg0)); // for empty file, remove should succeed
    EXPECT_FALSE(empty_db.FindRecordUnsafe(cfg0));  // no record in empty database
    EXPECT_FALSE(empty_db.StoreRecordUnsafe(cfg0)); // storing in an empty database should fail

    {
        miopen::TempFile temp_file("tmp-kerndb");
        miopen::KernDb clean_db(miopen::DbKinds::KernelDb, std::string(temp_file), false);

        EXPECT_TRUE(clean_db.StoreRecordUnsafe(cfg0));
        auto readout = clean_db.FindRecordUnsafe(cfg0);
        EXPECT_TRUE(readout);
        EXPECT_TRUE(readout.get() == cfg0.kernel_blob);
        EXPECT_TRUE(clean_db.RemoveRecordUnsafe(cfg0));
        EXPECT_FALSE(clean_db.FindRecordUnsafe(cfg0));
    }

    {
        miopen::TempFile temp_file("tmp-kerndb");
        miopen::KernDb err_db(
            miopen::DbKinds::KernelDb,
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
        EXPECT_TRUE(err_db.StoreRecordUnsafe(cfg0));
        // In which case decompresion should not be called
        EXPECT_TRUE(err_db.FindRecordUnsafe(cfg0));
        EXPECT_TRUE(err_db.RemoveRecordUnsafe(cfg0));
    }
}
#endif

TEST(TestCache, check_cache_file)
{
    auto p = miopen::GetCacheFile("gfx", "base", "args");
    EXPECT_TRUE(p.filename() == "base.o");
}
