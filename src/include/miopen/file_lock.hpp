// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef GUARD_MIOPEN_FILE_LOCK_HPP_
#define GUARD_MIOPEN_FILE_LOCK_HPP_

#include <chrono>
#include <string>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#include <errno.h>
#endif

#include <fcntl.h>

namespace miopen {

namespace detail {
template <class Mutex>
struct shareable_lock
{
    Mutex& m;
    shareable_lock(Mutex& m_) : m(m_) {}

    void lock() { m.lock_sharable(); }
    bool try_lock() { return m.try_lock_sharable(); }
};

template <typename Mutex>
bool try_timed_lock(Mutex& m, const std::chrono::time_point<std::chrono::steady_clock>& abs_time)
{
    if(m.try_lock())
    {
        return true;
    }
    while(std::chrono::steady_clock::now() < abs_time)
    {
        if(m.try_lock())
        {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}
} // namespace detail

class file_lock
{
public:
    file_lock() = default;

#ifdef _WIN32
    explicit file_lock(const std::string& filename)
    {
        // cppcheck-suppress useInitializationList
        handle = CreateFileA(filename.c_str(),
                             GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE,
                             nullptr,
                             OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL,
                             nullptr);

        if(handle == INVALID_HANDLE_VALUE)
            throw std::runtime_error("file_lock: cannot open file: " + filename);
    }
    ~file_lock()
    {
        if(handle != INVALID_HANDLE_VALUE)
            CloseHandle(handle);
    }
#else
    explicit file_lock(const std::string& filename)
    {
        fd = ::open(filename.c_str(), O_RDWR | O_CLOEXEC);
        if(fd == -1)
            throw std::runtime_error("file_lock: cannot open file: " + filename);
    }

    ~file_lock()
    {
        if(fd != -1)
            ::close(fd);
    }
#endif

    file_lock(const file_lock&) = delete;
    file_lock& operator=(const file_lock&) = delete;

    file_lock(file_lock&& rhs) noexcept { this->swap(rhs); }

    file_lock& operator=(file_lock&& rhs) noexcept
    {
        if(this != &rhs)
        {
            this->swap(rhs);
        }
        return *this;
    }

    void lock() { lock_impl(true, false); }

    bool try_lock() { return lock_impl(true, true); }

    bool timed_lock(const std::chrono::time_point<std::chrono::steady_clock>& abs_time)
    {
        return detail::try_timed_lock(*this, abs_time);
    }

#ifdef _WIN32
    void unlock() const
    {
        if(handle == INVALID_HANDLE_VALUE)
            return;

        OVERLAPPED ov = {};
        if(UnlockFileEx(handle, 0, MAXDWORD, MAXDWORD, &ov) == 0)
        {
            throw std::system_error(
                errno, std::generic_category(), "file_lock: failed to unlock file");
        }
    }
#else
    void unlock() const
    {
        if(fd == -1)
            return;

        flock fl    = {};
        fl.l_type   = F_UNLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start  = 0;
        fl.l_len    = 0;
        if(fcntl(fd, F_SETLK, &fl) != 0)
        {
            throw std::system_error(
                errno, std::generic_category(), "file_lock: failed to unlock file");
        }
    }
#endif

    void lock_sharable() { lock_impl(false, false); }

    bool try_lock_sharable() { return lock_impl(false, true); }

    bool timed_lock_sharable(const std::chrono::time_point<std::chrono::steady_clock>& abs_time)
    {
        detail::shareable_lock<file_lock> sfl(*this);
        return detail::try_timed_lock(sfl, abs_time);
    }

    void unlock_sharable() const { unlock(); }

private:
#ifdef _WIN32
    HANDLE handle{INVALID_HANDLE_VALUE};

    bool lock_impl(bool exclusive, bool nonblocking) const
    {
        DWORD flags = 0;
        if(exclusive)
            flags |= LOCKFILE_EXCLUSIVE_LOCK;
        if(nonblocking)
            flags |= LOCKFILE_FAIL_IMMEDIATELY;

        OVERLAPPED ov = {};
        BOOL ok       = LockFileEx(handle, flags, 0, MAXDWORD, MAXDWORD, &ov);
        return ok != 0;
    }

    void swap(file_lock& rhs) noexcept { std::swap(handle, rhs.handle); }
#else
    int fd{-1};

    bool lock_impl(bool exclusive, bool nonblocking) const
    {
        flock fl    = {};
        fl.l_type   = exclusive ? F_WRLCK : F_RDLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start  = 0;
        fl.l_len    = 0;
        int cmd     = nonblocking ? F_SETLK : F_SETLKW;
        int rc      = fcntl(fd, cmd, &fl);
        if(rc == -1)
        {
            if(nonblocking && (errno == EACCES || errno == EAGAIN))
                return false;
            throw std::system_error(
                errno, std::generic_category(), "file_lock: failed to lock file");
        }

        return true;
    }

    void swap(file_lock& rhs) noexcept { std::swap(fd, rhs.fd); }
#endif
};
} // namespace miopen

#endif // GUARD_MIOPEN_FILE_LOCK_HPP_
