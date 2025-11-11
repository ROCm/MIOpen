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

#include <miopen/errors.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/logger.hpp>

#include <boost/date_time/posix_time/posix_time_duration.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/time.hpp>
#include <boost/interprocess/sync/file_lock.hpp>

#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <thread>

namespace miopen {

class FSLockFile
{
public:
    FSLockFile() {}
    FSLockFile(const fs::path& path_) : path(path_)
    {
        lock_held     = false;
        lockfile_path = path.string() + ".fslock";
        unique_handle = lockfile_path.string() + "." + sysinfo::GetSystemHostname() + "." +
                        std::to_string(getpid());
    }

    bool timed_lock(const boost::posix_time::ptime& abs_time)
    {
        auto now      = boost::posix_time::second_clock::universal_time();
        bool acquired = false;
        MIOPEN_LOG_I2("Attempting Lock < " << lockfile_path.string());
        while(!acquired && now < abs_time)
        {
            now      = boost::posix_time::second_clock::universal_time();
            acquired = try_lock_hardlink();
            if(!acquired)
            {
                if(now < abs_time)
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            else
                MIOPEN_LOG_I2("Lock < " << lockfile_path.string());
        }
        return acquired;
    }

    void lock()
    {
        bool acquired = false;
        MIOPEN_LOG_I2("Attempting Lock < " << lockfile_path.string());
        while(!acquired)
        {
            acquired = try_lock_hardlink();
            if(!acquired)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            else
                MIOPEN_LOG_I2("Lock < " << lockfile_path.string());
        }
    }

    bool try_lock()
    {
        bool acquired = false;
        acquired      = try_lock_hardlink();
        if(acquired)
            MIOPEN_LOG_I2("Lock < " << lockfile_path.string());
        return acquired;
    }

    bool clear_stale_lock()
    {
        constexpr const auto timeout = std::chrono::milliseconds(20);
        auto last_write_time         = fs::last_write_time(lockfile_path);
        auto now                     = fs::file_time_type::clock::now();
        auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_write_time);
        if(age > timeout)
        {
            MIOPEN_LOG_I2("Removing Stale Lock < " << lockfile_path.string()
                                                   << ", Age(ms): " << age.count());
            fs::remove(lockfile_path);
            return true;
        }
        return false;
    }

    void refresh_lock()
    {
        constexpr const auto update_freq = std::chrono::milliseconds(4);
        MIOPEN_LOG_I2("Lock Refresh Active < " << unique_handle.string());
        auto last_refresh = fs::file_time_type::clock::now();
        auto age          = update_freq;
        while(lock_held)
        {
            if(age >= update_freq)
            {
                std::error_code ec;
                fs::last_write_time(unique_handle, fs::file_time_type::clock::now(), ec);
                if(ec.value() != 0)
                {
                    MIOPEN_LOG_I2("File <" << unique_handle << "> "
                                           << " time update failed. Terminating refresh."
                                           << "Error code: " << ec.value() << ". "
                                           << "Description: '" << ec.message() << "'");
                    return;
                }
                last_refresh = fs::file_time_type::clock::now();
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            auto now = fs::file_time_type::clock::now();
            age      = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_refresh);
        }
        MIOPEN_LOG_I2("Lock Refresh Exit < " << unique_handle.string());
    }

    bool try_lock_hardlink()
    {
        std::error_code ec;

        if(!fs::exists(unique_handle))
        {
            if(!std::ofstream{unique_handle})
                MIOPEN_THROW("Error creating file <" + unique_handle + "> for locking.");
            fs::permissions(unique_handle, FS_ENUM_PERMS_ALL);
        }

        create_hard_link(unique_handle, lockfile_path, ec);
        if(ec.value() == 0)
        {
            if(fs::hard_link_count(unique_handle) == 2)
            {
                lock_held      = true;
                refresh_thread = std::thread([this]() { this->refresh_lock(); });
                return true;
            }
        }
        else
        {
            clear_stale_lock();
        }

        return false;
    }

    void unlock()
    {
        MIOPEN_LOG_I2("Unlock < " << lockfile_path.string());
        lock_held = false;
        fs::remove(lockfile_path);
        fs::remove(unique_handle);
        refresh_thread.join();
    }

    fs::path get_unique_handle() { return unique_handle; }

private:
    bool lock_held;
    fs::path path;
    fs::path lockfile_path;
    fs::path unique_handle;
    std::thread refresh_thread;
};

MIOPEN_INTERNALS_EXPORT fs::path LockFilePath(const fs::path& filename_);
// LockFile class is a wrapper around boost::interprocess::file_lock providing MT-safety.
// One process should never have more than one instance of this class with same path at the same
// time. It may lead to undefined behaviour on Windows.
// Also on windows mutex can be removed because file locks are MT-safe there.
class MIOPEN_INTERNALS_EXPORT LockFile
{
private:
    class PassKey
    {
    };

public:
    LockFile(const fs::path&, PassKey);
    LockFile(const LockFile&) = delete;
    LockFile operator=(const LockFile&) = delete;

    bool timed_lock(const boost::posix_time::ptime& abs_time)
    {
        access_mutex.lock();
        bool ack = flock.timed_lock(abs_time);
        if(ack)
        {
            ack = fs_lock.timed_lock(abs_time);
            if(!ack)
                flock.unlock();
        }
        if(!ack)
            access_mutex.unlock();
        return ack;
    }

    bool timed_lock_shared(const boost::posix_time::ptime& abs_time)
    {
        access_mutex.lock_shared();
        return flock.timed_lock_sharable(abs_time);
    }
    void lock()
    {
        LockOperation(
            "lock", MIOPEN_GET_FN_NAME, [&]() { std::lock(access_mutex, flock, fs_lock); });
    }

    void lock_shared()
    {
        access_mutex.lock_shared();
        try
        {
            LockOperation("shared lock", MIOPEN_GET_FN_NAME, [&]() { flock.lock_sharable(); });
        }
        catch(...)
        {
            access_mutex.unlock();
        }
    }

    bool try_lock()
    {
        return TryLockOperation("lock", MIOPEN_GET_FN_NAME, [&]() {
            return std::try_lock(access_mutex, flock, fs_lock) == -1;
        });
    }

    bool try_lock_shared()
    {
        if(!access_mutex.try_lock_shared())
            return false;

        if(TryLockOperation(
               "shared lock", MIOPEN_GET_FN_NAME, [&]() { return flock.try_lock_sharable(); }))
            return true;
        access_mutex.unlock();
        return false;
    }

    void unlock()
    {
        LockOperation("fs_lock unlock", MIOPEN_GET_FN_NAME, [&]() { fs_lock.unlock(); });
        LockOperation("flock unlock", MIOPEN_GET_FN_NAME, [&]() { flock.unlock(); });
        access_mutex.unlock();
    }

    void unlock_shared()
    {
        LockOperation("unlock shared", MIOPEN_GET_FN_NAME, [&]() { flock.unlock_sharable(); });
        access_mutex.unlock_shared();
    }

    static LockFile& Get(const fs::path& file);

    template <class TDuration>
    bool try_lock_for(TDuration duration)
    {
        if(!access_mutex.try_lock_for(duration))
            return false;

        if(TryLockOperation("timed lock", MIOPEN_GET_FN_NAME, [&]() {
               bool ack = flock.timed_lock(ToPTime(duration));
               if(ack)
               {
                   ack = fs_lock.timed_lock(ToPTime(duration));
                   if(!ack)
                       flock.unlock();
               }
               return ack;
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

        if(TryLockOperation("shared timed lock", MIOPEN_GET_FN_NAME, [&]() {
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
    fs::path path; // For logging purposes
    std::shared_timed_mutex access_mutex;
    boost::interprocess::file_lock flock;
    FSLockFile fs_lock;

    static std::map<fs::path, LockFile>& LockFiles()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static std::map<fs::path, LockFile> lock_files;
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
                       const std::string_view from) const
    {
        // clang-format off
        MIOPEN_LOG_E_FROM(from, "File <" << path << "> " << operation << " failed. "
                                "Error code: " << ex.get_error_code() << ". "
                                "Native error: " << ex.get_native_error() << ". "
                                "Description: '" << ex.what() << "'");
        // clang-format on
    }

    void LockOperation(const std::string& op_name,
                       const std::string_view from,
                       std::function<void()>&& op)
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
                          const std::string_view from,
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
