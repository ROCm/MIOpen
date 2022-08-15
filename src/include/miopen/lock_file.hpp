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

#include <miopen/logger.hpp>

#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/time.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/interprocess/sync/file_lock.hpp>

#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace miopen {

std::string LockFilePath(const boost::filesystem::path& filename_);
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
    LockFile(const char* path_, PassKey);
    LockFile(const LockFile&) = delete;
    LockFile operator=(const LockFile&) = delete;

    bool timed_lock(const boost::posix_time::ptime& abs_time)
    {
        access_mutex.lock();
        return flock.timed_lock(abs_time);
    }

    bool timed_lock_shared(const boost::posix_time::ptime& abs_time)
    {
        access_mutex.lock_shared();
        return flock.timed_lock_sharable(abs_time);
    }
    void lock()
    {
        LockOperation("lock", MIOPEN_GET_FN_NAME(), [&]() { std::lock(access_mutex, flock); });
    }

    void lock_shared()
    {
        access_mutex.lock_shared();
        try
        {
            LockOperation("shared lock", MIOPEN_GET_FN_NAME(), [&]() { flock.lock_sharable(); });
        }
        catch(...)
        {
            access_mutex.unlock();
        }
    }

    bool try_lock()
    {
        return TryLockOperation("lock", MIOPEN_GET_FN_NAME(), [&]() {
            return std::try_lock(access_mutex, flock) != 0;
        });
    }

    bool try_lock_shared()
    {
        if(!access_mutex.try_lock_shared())
            return false;

        if(TryLockOperation(
               "shared lock", MIOPEN_GET_FN_NAME(), [&]() { return flock.try_lock_sharable(); }))
            return true;
        access_mutex.unlock();
        return false;
    }

    void unlock()
    {
        LockOperation("unlock", MIOPEN_GET_FN_NAME(), [&]() { flock.unlock(); });
        access_mutex.unlock();
    }

    void unlock_shared()
    {
        LockOperation("unlock shared", MIOPEN_GET_FN_NAME(), [&]() { flock.unlock_sharable(); });
        access_mutex.unlock_shared();
    }

    static LockFile& Get(const char* path);

    template <class TDuration>
    bool try_lock_for(TDuration duration)
    {
        if(!access_mutex.try_lock_for(duration))
            return false;

        if(TryLockOperation("timed lock", MIOPEN_GET_FN_NAME(), [&]() {
               return flock.timed_lock(ToPTime(duration));
           }))
            return true;
        access_mutex.unlock();
        return false;
    }

    template <class TDuration>
    bool try_lock_shared_for(TDuration duration)
    {
        if(!access_mutex.try_lock_shared_for(duration))
            return false;

        if(TryLockOperation("shared timed lock", MIOPEN_GET_FN_NAME(), [&]() {
               return flock.timed_lock_sharable(ToPTime(duration));
           }))
            return true;
        access_mutex.unlock();
        return false;
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
    const char* path; // For logging purposes
    std::shared_timed_mutex access_mutex;
    boost::interprocess::file_lock flock;

    static std::map<std::string, LockFile>& LockFiles()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
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

    void LogFlockError(const boost::interprocess::interprocess_exception& ex,
                       const std::string& operation,
                       const std::string& from) const
    {
        // clang-format off
        MIOPEN_LOG_E_FROM(from, "File <" << path << "> " << operation << " failed. "
                                "Error code: " << ex.get_error_code() << ". "
                                "Native error: " << ex.get_native_error() << ". "
                                "Description: '" << ex.what() << "'");
        // clang-format on
    }

    void
    LockOperation(const std::string& op_name, const std::string& from, std::function<void()>&& op)
    {
        try
        {
            op();
        }
        catch(const boost::interprocess::interprocess_exception& ex)
        {
            LogFlockError(ex, op_name, from);
            throw;
        }
    }

    bool TryLockOperation(const std::string& op_name,
                          const std::string& from,
                          std::function<bool()>&& op)
    {
        try
        {
            if(op())
                return true;
            MIOPEN_LOG_W("File <" << path << "> " << op_name << " timed out.");
            return false;
        }
        catch(const boost::interprocess::interprocess_exception& ex)
        {
            LogFlockError(ex, op_name, from);
            return false;
        }
    }
};
} // namespace miopen

#endif // GUARD_MIOPEN_LOCK_FILE_HPP_
