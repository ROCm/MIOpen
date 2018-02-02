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

#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/thread/shared_mutex.hpp>

namespace miopen {
// LockFile::Impl class is a wrapper around boost::interprocess::file_lock providing MT-safety.
// One process should never have more than one instance of this class with same path at the same
// time. It may lead to undefined behaviour on Windows.
// Also on windows mutex can be removed because file locks are MT-safe there.
// Dispatcher should be used to create instances of this class.
class LockFile
{
    friend class LockFileDispatcher;

    private:
    class Impl
    {
        public:
        Impl(const Impl&) = delete;
        Impl operator=(const Impl&) = delete;

        Impl(const char* path) : _file(path), _file_lock(path) {}

        void lock()
        {
            _mutex.lock();
            _file_lock.lock();
        }

        void lock_sharable()
        {
            _mutex.lock_shared();
            _file_lock.lock_sharable();
        }

        void unlock()
        {
            _file_lock.unlock();
            _mutex.unlock();
        }

        void unlock_sharable()
        {
            _file_lock.unlock_sharable();
            _mutex.unlock_shared();
        }

        private:
        boost::shared_mutex _mutex;
        std::ofstream _file;
        boost::interprocess::file_lock _file_lock;
    };

    public:
    LockFile(Impl& impl) : _impl(impl) {}

    void lock() { _impl.lock(); }
    void lock_sharable() { _impl.lock_sharable(); }
    void unlock() { _impl.unlock(); }
    void unlock_sharable() { _impl.unlock_sharable(); }

    private:
    Impl& _impl;
};

class LockFileDispatcher
{
    public:
    LockFileDispatcher() = delete;

    static LockFile Get(const char* path)
    {
        { // To guarantee that construction won't be called if not required.
            auto found = LockFiles().find(path);

            if(found != LockFiles().end())
                return {found->second};
        }

        auto emplaced = LockFiles().emplace(
            std::piecewise_construct, std::forward_as_tuple(path), std::forward_as_tuple(path));
        return {emplaced.first->second};
    }

    private:
    static std::map<std::string, LockFile::Impl>& LockFiles()
    {
        static std::map<std::string, LockFile::Impl> lock_files;
        return lock_files;
    }
};
} // namespace miopen

#endif // GUARD_MIOPEN_LOCK_FILE_HPP_
