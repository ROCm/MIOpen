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

#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <mutex>
#include <miopen/errors.hpp>
#include <miopen/config.h>

namespace miopen {

#define MIOPEN_DECLARE_HANDLE_MUTEX(x)                               \
    struct x                                                         \
    {                                                                \
        static const char* value() { return ".miopen-" #x ".lock"; } \
    };

#if MIOPEN_GPU_SYNC
MIOPEN_DECLARE_HANDLE_MUTEX(gpu_handle_mutex)
#define MIOPEN_HANDLE_LOCK \
    auto miopen_handle_lock_guard_##__LINE__ = miopen::get_handle_lock(miopen::gpu_handle_mutex{});
#else
#define MIOPEN_HANDLE_LOCK
#endif

inline boost::filesystem::path get_handle_lock_path(const char* name)
{
    auto p = boost::filesystem::current_path() / name;
    if(!boost::filesystem::exists(p))
    {
        auto tmp = boost::filesystem::current_path() / boost::filesystem::unique_path();
        boost::filesystem::ofstream{tmp};
        boost::filesystem::rename(tmp, p);
    }
    return p;
}

using handle_mutex = boost::interprocess::file_lock;
template <class T>
inline handle_mutex& get_handle_mutex(T)
{
    static handle_mutex m{get_handle_lock_path(T::value()).c_str()};
    return m;
}

using handle_lock = std::unique_lock<handle_mutex>;
template <class T>
inline handle_lock get_handle_lock(T, int timeout = 120)
{
    auto& m = get_handle_mutex(T{});
    if(m.timed_lock(boost::posix_time::second_clock::universal_time() +
                    boost::posix_time::seconds(timeout)))
    {
        return {m, std::adopt_lock_t{}};
    }
    else
    {
        m.unlock();
        MIOPEN_THROW("GPU is stalled");
    }
}

} // namespace miopen
