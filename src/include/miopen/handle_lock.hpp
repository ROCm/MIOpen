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

#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <mutex>
#include <miopen/errors.hpp>

namespace miopen {

#if MIOPEN_BUILD_DEV

#if 0
using handle_mutex = boost::interprocess::named_mutex;
inline handle_mutex& get_handle_mutex()
{
    static handle_mutex m{boost::interprocess::open_or_create, "miopen_gpu_handle_lock"};
    return m;
}
#else
inline boost::filesystem::path get_handle_lock_path()
{
    auto tmp = boost::filesystem::current_path() / boost::filesystem::unique_path();
    auto p   = boost::filesystem::current_path() / ".miopen-gpu-handle.lock";
    boost::filesystem::ofstream{tmp};
    boost::filesystem::rename(tmp, p);
    return p;
}

using handle_mutex = boost::interprocess::file_lock;
inline handle_mutex& get_handle_mutex()
{
    static handle_mutex m{get_handle_lock_path().c_str()};
    return m;
}
#endif
using handle_lock = std::unique_lock<handle_mutex>;
inline handle_lock get_handle_lock()
{
    auto& m = get_handle_mutex();
    if(m.timed_lock(boost::posix_time::second_clock::local_time() +
                    boost::posix_time::seconds(120)))
    {
        return {m, std::adopt_lock_t{}};
    }
    else
    {
        m.unlock();
        MIOPEN_THROW("GPU is stalled");
    }
}

#else

struct handle_lock
{
};
inline handle_lock get_handle_lock() { return {}; }

#endif

} // namespace miopen
