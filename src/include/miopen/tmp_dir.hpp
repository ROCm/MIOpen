/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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
#ifndef MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP
#define MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP

#include <string>
#include <boost/filesystem/path.hpp>

namespace miopen {

void SystemCmd(std::string cmd);

struct TmpDir
{
    boost::filesystem::path path;
    TmpDir(std::string prefix);

    TmpDir(TmpDir const&) = delete;
    TmpDir& operator=(TmpDir const&) = delete;

    TmpDir(TmpDir&& other) noexcept { (*this) = std::move(other); }
    TmpDir& operator=(TmpDir&& other) noexcept;

    void Execute(std::string exe, std::string args) const;

    ~TmpDir();
};

} // namespace miopen

#endif
