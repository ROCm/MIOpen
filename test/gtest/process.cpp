/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <algorithm>
#include <array>
#include <climits>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include <miopen/env.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/process.hpp>
#include <miopen/tmp_dir.hpp>

#include "gtest/gtest.h"

namespace fs  = miopen::fs;
namespace env = miopen::env;

#ifndef _WIN32
#include <cstring>
#else
#include <io.h>
#include <fcntl.h>
#endif

namespace {
constexpr std::string_view string_data =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
    "sed do eiusmod tempor incididunt ut labore et dolore magna "
    "aliqua. Ut enim ad minim veniam, quis nostrud exercitation "
    "ullamco laboris nisi ut aliquip ex ea commodo consequat. "
    "Duis aute irure dolor in reprehenderit in voluptate velit "
    "esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
    "occaecat cupidatat non proident, sunt in culpa qui officia "
    "deserunt mollit anim id est laborum.";

fs::path executable{}; // NOLINT

void write_buffer(const fs::path& filename, const std::vector<char>& buffer)
{
    std::ofstream os(filename, std::ios::out | std::ios::binary);
    os.write(buffer.data(), buffer.size());
}

template <typename T>
T generic_read_file(const fs::path& filename, const size_t offset = 0, size_t nbytes = 0)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if(!is.is_open())
        MIOPEN_THROW("Failure opening file: " + filename);
    if(nbytes == 0)
    {
        // if there is a non-zero offset and nbytes is not set,
        // calculate size of remaining bytes to read
        nbytes = is.tellg();
        if(offset > nbytes)
            MIOPEN_THROW("offset is larger than file size");
        nbytes -= offset;
    }
    if(nbytes < 1)
        MIOPEN_THROW("Invalid size for: " + filename);
    is.seekg(offset, std::ios::beg);

    T buffer(nbytes, 0);
    if(!is.read(&buffer[0], nbytes))
        MIOPEN_THROW("Error reading file: " + filename);
    return buffer;
}

std::vector<char>
read_buffer(const fs::path& filename, const size_t offset = 0, const size_t nbytes = 0)
{
    return generic_read_file<std::vector<char>>(filename, offset, nbytes);
}

std::string read_string(const fs::path& filename)
{
    return generic_read_file<std::string>(filename);
}

std::vector<char> read_stdin()
{
    std::vector<char> result;
    std::array<char, 1024> buffer{};
    std::size_t len = 0;
#ifdef _WIN32
    // Set stream mode to BINARY to suppress translations.
    // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmode?view=msvc-170
    auto old_mode = _setmode(_fileno(stdin), _O_BINARY);
    EXPECT_NE(old_mode, -1);
#endif
    while((len = std::fread(buffer.data(), 1, buffer.size(), stdin)) > 0)
    {
        if(std::ferror(stdin) != 0 and std::feof(stdin) == 0)
            throw std::runtime_error{std::strerror(errno)};

        result.insert(result.end(), buffer.begin(), buffer.begin() + len);
    }
#ifdef _WIN32
    // Reset to the previously set translation mode.
    _setmode(_fileno(stdin), old_mode);
#endif
    return result;
}
} // namespace

TEST(ProcessTest, StringStdin)
{
    const miopen::TmpDir tmp{};
    const auto out = tmp / "output.txt";
    miopen::Process p{executable, "--stdin " + out};
    EXPECT_EQ(p.Write(string_data).Wait(), 0);
    EXPECT_EQ(fs::is_regular_file(out), true);
    const std::string result{read_string(out)};
    EXPECT_EQ(result, string_data);
    EXPECT_EQ(fs::remove(out), true);
}

TEST(ProcessTest, BinaryStdin)
{
    std::random_device rd;
    std::independent_bits_engine<std::mt19937, CHAR_BIT, unsigned short> rbe(rd());
    std::vector<char> binary_data(4096);
    std::generate(binary_data.begin(), binary_data.end(), std::ref(rbe));
    const miopen::TmpDir tmp{};
    const auto out = tmp / "output.bin";
    miopen::Process p{executable, "--stdin " + out};
    EXPECT_EQ(p.Write(binary_data).Wait(), 0);
    EXPECT_EQ(fs::is_regular_file(out), true);
    std::vector<char> result{read_buffer(out)};
    EXPECT_EQ(result, binary_data);
    EXPECT_EQ(fs::remove(out), true);
}

TEST(ProcessTest, ReadStdout)
{
    std::string buffer;
    miopen::Process p{executable, "--stdout"};
    EXPECT_EQ(p.Read(buffer).Wait(), 0);
    EXPECT_EQ(buffer, string_data);
}

TEST(ProcessTest, CurrentWorkingDirectory)
{
    const miopen::TmpDir tmp{};
    const auto out = tmp / "output.txt";
    miopen::Process p{executable, "--stdin output.txt"};
    EXPECT_EQ(p.WorkingDirectory(tmp).Write(string_data).Wait(), 0);
    EXPECT_EQ(fs::is_regular_file(out), true);
    const std::string result{read_string(out)};
    EXPECT_EQ(result, string_data);
    EXPECT_EQ(fs::remove(out), true);
}

TEST(ProcessTest, EnvironmentVariable)
{
    std::string buffer;
    miopen::Process p{executable, "--stdout"};
    p.EnvironmentVariables({{"MIOPEN_PROCESS_TEST_ENVIRONMENT_VARIABLE", "1"}});
    EXPECT_EQ(p.Read(buffer).Wait(), 0);
    std::string reversed{string_data.begin(), string_data.end()};
    std::reverse(reversed.begin(), reversed.end());
    EXPECT_EQ(buffer, reversed);
};

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_PROCESS_TEST_ENVIRONMENT_VARIABLE)

int main(int argc, char* argv[])
{
    if(argc > 1)
    {
        const std::string_view arg{argv[1]};
        if(arg == "--stdin")
        {
            write_buffer(argv[2], read_stdin());
            return 0;
        }
        if(arg == "--stdout")
        {
            std::vector<char> result{string_data.begin(), string_data.end()};
            if(env::enabled(MIOPEN_PROCESS_TEST_ENVIRONMENT_VARIABLE))
                std::reverse(result.begin(), result.end());
            std::fwrite(result.data(), 1, result.size(), stdout);
            return 0;
        }
    }

    executable = argv[0];
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
