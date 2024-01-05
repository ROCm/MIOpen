/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_FILESYSTEM_HPP_
#define GUARD_MIOPEN_FILESYSTEM_HPP_

#if defined(CPPCHECK)
#define MIOPEN_HAS_FILESYSTEM 1
#define MIOPEN_HAS_FILESYSTEM_TS 1
#elif defined(_WIN32)
#if _MSC_VER >= 1920
#define MIOPEN_HAS_FILESYSTEM 1
#define MIOPEN_HAS_FILESYSTEM_TS 0
#elif _MSC_VER >= 1900
#define MIOPEN_HAS_FILESYSTEM 0
#define MIOPEN_HAS_FILESYSTEM_TS 1
#else
#define MIOPEN_HAS_FILESYSTEM 0
#define MIOPEN_HAS_FILESYSTEM_TS 0
#endif
#elif defined(__has_include)
#if __has_include(<filesystem>) && __cplusplus >= 201703L
#define MIOPEN_HAS_FILESYSTEM 1
#else
#define MIOPEN_HAS_FILESYSTEM 0
#endif
#if __has_include(<experimental/filesystem>) && __cplusplus >= 201103L
#define MIOPEN_HAS_FILESYSTEM_TS 1
#else
#define MIOPEN_HAS_FILESYSTEM_TS 0
#endif
#else
#define MIOPEN_HAS_FILESYSTEM 0
#define MIOPEN_HAS_FILESYSTEM_TS 0
#endif

#if MIOPEN_HAS_FILESYSTEM
#include <filesystem>
#elif MIOPEN_HAS_FILESYSTEM_TS
#include <experimental/filesystem>
#else
#error "No filesystem include available"
#endif

namespace miopen {

#if MIOPEN_HAS_FILESYSTEM
namespace fs = ::std::filesystem;
#elif MIOPEN_HAS_FILESYSTEM_TS
namespace fs = ::std::experimental::filesystem;
#endif

inline std::string operator+(const std::string_view s, const fs::path& p)
{
    return p.string().insert(0, s);
}

inline std::string operator+(const fs::path& p, const std::string_view s)
{
    return p.string().append(s);
}

inline fs::path append_extension(const fs::path& p, const std::string_view s)
{
    return fs::path{p}.replace_extension(p.extension().string().append(s));
}

} // namespace miopen

#if MIOPEN_HAS_FILESYSTEM_TS
#ifdef __linux__
#include <linux/limits.h>
namespace miopen {
inline fs::path weakly_canonical(const fs::path& path)
{
    std::string result(PATH_MAX, '\0');
    std::string p{path.is_relative() ? (fs::current_path() / path).string() : path.string()};
    char* retval = realpath(p.c_str(), &result[0]);
    return (retval == nullptr) ? path : fs::path{result};
}
} // namespace miopen
#else
#error "Not implmeneted!"
#endif
#else
namespace miopen {
inline fs::path weakly_canonical(const fs::path& path) { return fs::weakly_canonical(path); }
} // namespace miopen
#endif

#endif // GUARD_MIOPEN_FILESYSTEM_HPP_
