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
#include <miopen/lock_file.hpp>

#include <boost/date_time/posix_time/posix_time_types.hpp>

#include <mutex>

namespace miopen {

// void LockFile::lock() { std::lock(_mutex, _file_lock); }

// void LockFile::lock_shared()
// {
//     _mutex.lock_shared();
//     _file_lock.lock_sharable();
// }

// bool LockFile::try_lock() { return std::try_lock(_mutex, _file_lock); }

// bool LockFile::try_lock_shared()
// {
//     if(!_mutex.try_lock_shared())
//         return false;

//     if(_file_lock.try_lock_sharable())
//         return true;

//     _mutex.unlock_shared();
//     return false;
// }

// void LockFile::unlock()
// {
//     _file_lock.unlock();
//     _mutex.unlock();
// }

// void LockFile::unlock_shared()
// {
//     _file_lock.unlock_sharable();
//     _mutex.unlock_shared();
// }

LockFile& LockFile::Get(const char* path)
{
    { // To guarantee that construction won't be called if not required.
        auto found = LockFiles().find(path);

        if(found != LockFiles().end())
            return found->second;
    }

    auto emplaced = LockFiles().emplace(std::piecewise_construct,
                                        std::forward_as_tuple(path),
                                        std::forward_as_tuple(path, PassKey{}));
    return emplaced.first->second;
}
} // namespace miopen
