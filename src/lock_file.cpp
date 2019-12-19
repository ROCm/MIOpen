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
#include <miopen/md5.hpp>

namespace miopen {

std::string LockFilePath(const boost::filesystem::path& filename_)
{
    const auto directory = boost::filesystem::temp_directory_path() / "miopen-lockfiles";

    if(!exists(directory))
    {
        boost::filesystem::create_directories(directory);
        boost::filesystem::permissions(directory, boost::filesystem::all_all);
    }
    const auto hash = md5(filename_.parent_path().string());
    const auto file = directory / (hash + "_" + filename_.filename().string() + ".lock");

    return file.string();
}

LockFile& LockFile::Get(const char* path)
{
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

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
