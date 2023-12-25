/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_GUARD_MLOPEN_PROCESS_HPP
#define MIOPEN_GUARD_MLOPEN_PROCESS_HPP

#include <boost/filesystem.hpp>
#include <memory>
#include <string_view>

namespace miopen {

struct ProcessImpl;

struct Process
{
    Process(const boost::filesystem::path& cmd);
    ~Process() noexcept;

    int operator()(std::string_view args = "", const boost::filesystem::path& cwd = "");

private:
    std::unique_ptr<ProcessImpl> impl;
};

struct ProcessAsync
{
    ProcessAsync(const boost::filesystem::path& cmd,
                 std::string_view args              = "",
                 const boost::filesystem::path& cwd = "");
    ~ProcessAsync() noexcept;

    ProcessAsync(ProcessAsync&&) noexcept;
    ProcessAsync& operator=(ProcessAsync&&) noexcept;

    int Wait();

private:
    std::unique_ptr<ProcessImpl> impl;
};

} // namespace miopen

#endif // MIOPEN_GUARD_MLOPEN_PROCESS_HPP
