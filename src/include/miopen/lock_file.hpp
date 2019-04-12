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
#ifndef GUARD_MIOPEN_LOCK_FILE_HPP_
#define GUARD_MIOPEN_LOCK_FILE_HPP_

#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/time.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/interprocess/sync/file_lock.hpp>

#include <chrono>
#include <fstream>
#include <shared_mutex>
#include <map>
#include <mutex>
#include <string>

namespace miopen {
// LockFile class is a wrapper around boost::interprocess::file_lock providing MT-safety.
// One process should never have more than one instance of this class with same path at the same
// time. It may lead to undefined behaviour on Windows.
// Also on windows mutex can be removed because file locks are MT-safe there.
class LockFile
{
    private:
    class PassKey
    {
    };

    public:
    LockFile(const char* path, PassKey)
    {
        if(!boost::filesystem::exists(path))
        {
            std::ofstream{path};
            boost::filesystem::permissions(path, boost::filesystem::all_all);
        }
        flock = path;
    }
    LockFile(const LockFile&) = delete;
    LockFile operator=(const LockFile&) = delete;

    void lock() { std::lock(access_mutex, flock); }
    void lock_shared()
    {
        access_mutex.lock_shared();
        flock.lock_sharable();
    }
    bool try_lock() { return std::try_lock(access_mutex, flock) != 0; }
    bool try_lock_shared()
    {
        if(!access_mutex.try_lock_shared())
            return false;

        if(!flock.try_lock_sharable())
        {
            access_mutex.unlock_shared();
            return false;
        }
        return true;
    }
    void unlock()
    {
        flock.unlock();
        access_mutex.unlock();
    }
    void unlock_shared()
    {
        flock.unlock_sharable();
        access_mutex.unlock_shared();
    }

    static LockFile& Get(const char* path);

    template <class TDuration>
    bool try_lock_for(TDuration duration)
    {
        if(!access_mutex.try_lock_for(duration))
            return false;

        if(!flock.timed_lock(ToPTime(duration)))
        {
            access_mutex.unlock();
            return false;
        }
        return true;
    }

    template <class TDuration>
    bool try_lock_shared_for(TDuration duration)
    {
        if(!access_mutex.try_lock_shared_for(duration))
            return false;

        if(!flock.timed_lock_sharable(ToPTime(duration)))
        {
            access_mutex.unlock();
            return false;
        }
        return true;
    }

    template <class TPoint>
    bool try_lock_until(TPoint point)
    {
        return try_lock_for(point - std::chrono::system_clock::now());
    }

    template <class TPoint>
    bool try_lock_shared_until(TPoint point)
    {
        return try_lock_shared_for(point - std::chrono::system_clock::now());
    }

    private:
    std::shared_timed_mutex access_mutex;
    boost::interprocess::file_lock flock;

    static std::map<std::string, LockFile>& LockFiles()
    {
        static std::map<std::string, LockFile> lock_files;
        return lock_files;
    }

    template <class TDuration>
    static boost::posix_time::ptime ToPTime(TDuration duration)
    {
        return boost::posix_time::second_clock::universal_time() +
               boost::posix_time::milliseconds(
                   std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    }
};
} // namespace miopen

#endif // GUARD_MIOPEN_LOCK_FILE_HPP_
