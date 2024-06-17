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

#include <miopen/config.hpp>
#include <miopen/filesystem.hpp>
#include <memory>
#include <string_view>
#include <map>

namespace miopen {

struct ProcessImpl;

struct MIOPEN_INTERNALS_EXPORT Process
{
    explicit Process(const fs::path& cmd);
    ~Process() noexcept;

    Process(Process&&) noexcept;
    Process& operator=(Process&&) noexcept;

    Process& Arguments(std::string_view args);
    Process& WorkingDirectory(const fs::path& cwd);
    Process& EnvironmentVariables(std::map<std::string_view, std::string_view> vars);

    const Process& Capture(std::vector<char>& buffer) const;
    const Process& Execute(const std::vector<char>& buffer = {}) const;

    int Wait() const;

private:
    std::unique_ptr<ProcessImpl> impl;
};

} // namespace miopen

#endif // MIOPEN_GUARD_MLOPEN_PROCESS_HPP
