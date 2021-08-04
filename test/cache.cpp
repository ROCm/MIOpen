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
#include "test.hpp"
#include "random.hpp"

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

int main()
{
    check_cache_file();
    check_cache_str();
#if MIOPEN_ENABLE_SQLITE
    check_bz2_compress();
    check_bz2_decompress();
    check_kern_db();
#endif
}
