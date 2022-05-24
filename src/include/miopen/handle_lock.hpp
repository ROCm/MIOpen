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

#ifndef GUARD_MIOPEN_HANDLE_LOCK_HPP
#define GUARD_MIOPEN_HANDLE_LOCK_HPP

#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <mutex>
#include <miopen/config.h>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>

namespace miopen {

#define MIOPEN_DECLARE_HANDLE_MUTEX(x)                               \
    struct x                                                         \
    {                                                                \
        static const char* value() { return ".miopen-" #x ".lock"; } \
    };

#if MIOPEN_GPU_SYNC
MIOPEN_DECLARE_HANDLE_MUTEX(gpu_handle_mutex)
#define MIOPEN_HANDLE_LOCK                                    \
    auto MIOPEN_PP_CAT(miopen_handle_lock_guard_, __LINE__) = \
        miopen::get_handle_lock(miopen::gpu_handle_mutex{});
#else
#define MIOPEN_HANDLE_LOCK
#endif

inline boost::filesystem::path get_handle_lock_path(const char* name)
{
    auto p = boost::filesystem::current_path() / name;
    if(!boost::filesystem::exists(p))
    {
        auto tmp = boost::filesystem::current_path() / boost::filesystem::unique_path();
        boost::filesystem::ofstream{tmp}; // NOLINT
        boost::filesystem::rename(tmp, p);
    }
    return p;
}

struct handle_mutex
{
    std::recursive_timed_mutex m;
    boost::interprocess::file_lock flock;

    handle_mutex(const char* name) : flock(name) {}

    bool try_lock() { return std::try_lock(m, flock) != 0; }

    void lock() { std::lock(m, flock); }

    template <class Duration>
    bool try_lock_for(Duration d)
    {
        return m.try_lock_for(d) &&
               flock.timed_lock(
                   boost::posix_time::second_clock::universal_time() +
                   boost::posix_time::milliseconds(
                       std::chrono::duration_cast<std::chrono::milliseconds>(d).count()));
    }

    template <class Point>
    bool try_lock_until(Point p)
    {
        return m.try_lock_for(p - std::chrono::system_clock::now());
    }

    void unlock()
    {
        flock.unlock();
        m.unlock();
    }
};

template <class T>
inline std::unique_lock<handle_mutex> get_handle_lock(T, int timeout = 120)
{
    // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
    static handle_mutex m{get_handle_lock_path(T::value()).c_str()};
    return {m, std::chrono::seconds{timeout}};
}

} // namespace miopen

#endif // GUARD_MIOPEN_HANDLE_LOCK_HPP
