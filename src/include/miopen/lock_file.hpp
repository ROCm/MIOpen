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

#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/thread/shared_mutex.hpp>

namespace boost {
namespace posix_time {
class ptime;
} // namespace posix_time
} // namespace boost

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
    LockFile(const char* path, PassKey) : _file(path), _file_lock(path) {}
    LockFile(const LockFile&) = delete;
    LockFile operator=(const LockFile&) = delete;

    void lock();
    void lock_sharable();
    bool timed_lock(const boost::posix_time::ptime& abs_time);
    bool timed_lock_sharable(const boost::posix_time::ptime& abs_time);
    void unlock();
    void unlock_sharable();

    static LockFile& Get(const char* path);

    private:
    boost::shared_mutex _mutex;
    std::ofstream _file;
    boost::interprocess::file_lock _file_lock;

    static std::map<std::string, LockFile>& LockFiles()
    {
        static std::map<std::string, LockFile> lock_files;
        return lock_files;
    }
};
} // namespace miopen

#endif // GUARD_MIOPEN_LOCK_FILE_HPP_
