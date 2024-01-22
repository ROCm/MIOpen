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

#include <miopen/errors.hpp>
#include <miopen/process.hpp>
#include <string_view>

namespace miopen {

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

struct ProcessImpl
{
public:
    ProcessImpl(std::string_view cmd) : path{cmd} {}

    void Create(std::string_view args, std::string_view cwd)
    {
        STARTUPINFOA info;
        ZeroMemory(&info, sizeof(STARTUPINFO));
        info.cb = sizeof(STARTUPINFO);

        std::string cmd{path.string()};
        if(!args.empty())
            cmd += " " + std::string{args};

        // Refer to
        // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessa
        constexpr std::size_t BUFFER_CAPACITY = 32767;

        TCHAR buffer[BUFFER_CAPACITY];
        std::strncpy(buffer, cmd.data(), BUFFER_CAPACITY);

        if(CreateProcess(path.string().c_str(),
                         buffer,
                         nullptr,
                         nullptr,
                         FALSE,
                         0,
                         nullptr,
                         cwd.empty() ? nullptr : cwd.data(),
                         &info,
                         &processInfo) == FALSE)
            MIOPEN_THROW("CreateProcess error: " + std::to_string(GetLastError()));
    }

    int Wait()
    {
        WaitForSingleObject(processInfo.hProcess, INFINITE);

        DWORD status;
        const auto getExitCodeStatus = GetExitCodeProcess(processInfo.hProcess, &status);

        CloseHandle(processInfo.hProcess);
        CloseHandle(processInfo.hThread);

        if(getExitCodeStatus == 0)
            MIOPEN_THROW("GetExitCodeProcess error: " + std::to_string(GetLastError()));

        return status;
    }

private:
    fs::path path;
    PROCESS_INFORMATION processInfo{};
};

#else

struct ProcessImpl
{
    ProcessImpl(std::string_view cmd) : path{cmd} {}

    void Create(std::string_view args, std::string_view cwd)
    {
        std::string cmd{path.string()};
        if(!args.empty())
            cmd += " " + std::string{args};
        if(!cwd.empty())
            cmd.insert(0, "cd " + std::string{cwd} + "; ");
        pipe = popen(cmd.c_str(), "w");
        if(pipe == nullptr)
            MIOPEN_THROW("Error: popen()");
    }

    int Wait()
    {
        auto status = pclose(pipe);
        return WEXITSTATUS(status);
    }

private:
    fs::path path;
    FILE* pipe = nullptr;
};

#endif

Process::Process(const fs::path& cmd)
    : impl{std::make_unique<ProcessImpl>(cmd.string())}
{
}

Process::~Process() noexcept = default;

int Process::operator()(std::string_view args, const fs::path& cwd)
{
    impl->Create(args, cwd.string());
    return impl->Wait();
}

ProcessAsync::ProcessAsync(const fs::path& cmd,
                           std::string_view args,
                           const fs::path& cwd)
    : impl{std::make_unique<ProcessImpl>(cmd.string())}
{
    impl->Create(args, cwd.string());
}

ProcessAsync::~ProcessAsync() noexcept = default;

int ProcessAsync::Wait() { return impl->Wait(); }

ProcessAsync& ProcessAsync::operator=(ProcessAsync&& other) noexcept
{
    impl = std::move(other.impl);
    return *this;
}

ProcessAsync::ProcessAsync(ProcessAsync&& other) noexcept : impl{std::move(other.impl)} {}

} // namespace miopen
