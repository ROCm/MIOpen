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

#ifndef GUARD_TEMP_FILE_HPP
#define GUARD_TEMP_FILE_HPP

#include <miopen/tmp_dir.hpp>

#include <string>

namespace miopen {

class TempFile
{
    public:
    TempFile(const std::string& path_template);

    TempFile(TempFile&& other) noexcept : name(std::move(other.name)), dir(std::move(other.dir)) {}

    TempFile& operator=(TempFile&& other) noexcept
    {
        name = std::move(other.name);
        dir  = std::move(other.dir);
        return *this;
    }

    const std::string& Name() const { return name; }
    std::string Path() const { return (dir.path / name).string(); }
    operator std::string() const { return Path(); }

    private:
    std::string name;
    TmpDir dir;
};
} // namespace miopen

#endif
