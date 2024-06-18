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
#include <functional>
#include <vector>

#ifdef _WIN32
#include <miopen/tmp_dir.hpp>
#include <sstream>
#include <system_error>
#endif

namespace miopen {

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

constexpr std::size_t MIOPEN_PROCESS_BUFSIZE = 4096;

enum class Direction : bool
{
    Input,
    Output
};

struct SystemError : std::runtime_error
{
    SystemError() : std::runtime_error{std::system_category().message(GetLastError())} {}
};

template <Direction direction>
struct Pipe
{
    HANDLE readHandle, writeHandle;

    Pipe() : readHandle{nullptr}, writeHandle{nullptr}
    {
        SECURITY_ATTRIBUTES attrs;
        attrs.nLength              = sizeof(SECURITY_ATTRIBUTES);
        attrs.bInheritHandle       = TRUE;
        attrs.lpSecurityDescriptor = nullptr;

        if(CreatePipe(&readHandle, &writeHandle, &attrs, 0) == FALSE)
            throw SystemError();

        if(direction == Direction::Output)
        {
            // Do not inherit the read handle for the output pipe
            if(SetHandleInformation(readHandle, HANDLE_FLAG_INHERIT, 0) == 0)
                throw SystemError();
        }
        else
        {
            // Do not inherit the write handle for the input pipe
            if(SetHandleInformation(writeHandle, HANDLE_FLAG_INHERIT, 0) == 0)
                throw SystemError();
        }
    }

    Pipe(Pipe&&) = default;

    ~Pipe()
    {
        if(writeHandle != nullptr)
        {
            CloseHandle(writeHandle);
        }
        if(readHandle != nullptr)
        {
            CloseHandle(readHandle);
        }
    }

    bool CloseWriteHandle()
    {
        auto result = true;
        if(writeHandle != nullptr)
        {
            result      = CloseHandle(writeHandle) == TRUE;
            writeHandle = nullptr;
        }
        return result;
    }

    bool CloseReadHandle()
    {
        auto result = true;
        if(readHandle != nullptr)
        {
            result     = CloseHandle(readHandle) == TRUE;
            readHandle = nullptr;
        }
        return result;
    }

    std::pair<bool, DWORD> Read(void* buffer, DWORD size) const
    {
        DWORD bytes_read;
        if(ReadFile(readHandle, buffer, size, &bytes_read, nullptr) == FALSE &&
           GetLastError() == ERROR_MORE_DATA)
        {
            return {true, bytes_read};
        }
        return {false, bytes_read};
    }

    bool Write(const void* buffer, DWORD size) const
    {
        DWORD bytes_written;
        return WriteFile(writeHandle, buffer, size, &bytes_written, nullptr) == TRUE;
    }
};

struct ProcessImpl
{
    ProcessImpl(std::string_view cmd, std::string_view arguments) : command{cmd}, args{arguments} {}

    void Execute()
    {
        // See CreateProcess() WIN32 documentation for details.
        constexpr std::size_t CMDLINE_LENGTH = 32767;

        // Build lpCommandLine parameter.
        std::string cmdline{command};
        if(!args.empty())
            cmdline += " " + args;

        // clang-format off
        if(cmdline.size() > CMDLINE_LENGTH)
            MIOPEN_THROW("Command line too long, required maximum " +
                         std::to_string(CMDLINE_LENGTH) + " cjaracters.");
        // clang-format on

        if(cmdline.size() < CMDLINE_LENGTH)
            cmdline.resize(CMDLINE_LENGTH, '\0');

        STARTUPINFOA info;
        ZeroMemory(&info, sizeof(STARTUPINFO));
        info.cb         = sizeof(STARTUPINFO);
        info.hStdError  = output.writeHandle;
        info.hStdOutput = output.writeHandle;
        info.hStdInput  = input.readHandle;
        info.dwFlags |= STARTF_USESTDHANDLES;

        ZeroMemory(&processInfo, sizeof(PROCESS_INFORMATION));

        if(CreateProcess(command.c_str(),
                         cmdline.data(),
                         nullptr,
                         nullptr,
                         TRUE,
                         0,
                         // lpEnvironment,
                         envs.empty() ? nullptr : envs.data(),
                         cwd.empty() ? nullptr : cwd.c_str(),
                         &info,
                         &processInfo) == FALSE)
            MIOPEN_THROW("CreateProcess error: " + std::to_string(GetLastError()));

        CloseHandle(processInfo.hThread);

        if(!output.CloseWriteHandle())
            MIOPEN_THROW("Error closing STDOUT handle for writing (" +
                         std::to_string(GetLastError()) + ")");

        if(!input.CloseReadHandle())
            MIOPEN_THROW("Error closing STDIN handle for reading (" +
                         std::to_string(GetLastError()) + ")");
    }

    void EnvironmentVariables(const std::map<std::string_view, std::string_view>& vars)
    {
        envs.clear();
        auto envStrings = GetEnvironmentStrings();
        if(envStrings == nullptr)
            MIOPEN_THROW("Unable to get environment strings");
        auto p = envStrings;
        while(*p != 0)
        {
            while(*p != 0)
            {
                envs.push_back(*p++);
            }
            envs.push_back(*p++);
        }
        for(const auto& [name, value] : vars)
        {
            envs.insert(envs.end(), name.begin(), name.end());
            envs.push_back('=');
            envs.insert(envs.end(), value.begin(), value.end());
            envs.push_back('\0');
        }
        envs.push_back('\0');
        FreeEnvironmentStrings(envStrings);
    }

    void WorkingDirectory(const fs::path& path) { cwd = path.string(); }

    template <typename T>
    void Read(T& buffer)
    {
        Execute();
        TCHAR chunk[MIOPEN_PROCESS_BUFSIZE];
        for(;;)
        {
            auto [more_data, bytes_read] = output.Read(chunk, MIOPEN_PROCESS_BUFSIZE);
            if(bytes_read == 0)
                break;
            buffer.insert(buffer.end(), chunk, chunk + bytes_read);
            if(!more_data)
                break;
        }
    }

    void Write(const void* buffer, const std::size_t size)
    {
        Execute();
        std::ignore = input.Write(buffer, size);
    }

    int Wait() const
    {
        if(!input.CloseWriteHandle())
            MIOPEN_THROW("Error closing STDIN handle for writing (" +
                         std::to_string(GetLastError()) + ")");

        WaitForSingleObject(processInfo.hProcess, INFINITE);

        DWORD status{};
        GetExitCodeProcess(processInfo.hProcess, &status);

        CloseHandle(processInfo.hProcess);

        return static_cast<int>(status);
    }

private:
    std::string command;
    PROCESS_INFORMATION processInfo{};
    std::string args;
    std::string cwd;
    std::vector<CHAR> envs;
    mutable Pipe<Direction::Input> input;
    Pipe<Direction::Output> output;
};

#else

struct ProcessImpl
{
    ProcessImpl(std::string_view cmd, std::string_view arguments) : command{cmd}, args{arguments} {}

    std::string GetCommand() const
    {
        std::string cmd;
        if(!cwd.empty())
            cmd += "cd " + cwd + "; ";
        cmd += envs + command;
        if(!args.empty())
            cmd += " " + args;
        return cmd;
    }

    void Execute()
    {
        pipe = popen(GetCommand().c_str(), "r");
        if(pipe == nullptr)
            MIOPEN_THROW("Error: popen()");
    }

    template <typename T>
    void Read(T& buffer)
    {
        Execute();
        std::array<char, 128> chunk;
        buffer.clear();
        while(fgets(chunk.data(), chunk.size(), pipe) != nullptr)
            buffer.insert(buffer.end(), chunk.data(), chunk.data() + strlen(chunk.data()));
    }

    void Write(const void* buffer, const std::size_t size)
    {
        pipe = popen(GetCommand().c_str(), "w");
        if(pipe == nullptr)
            MIOPEN_THROW("Error: popen()");
        std::fwrite(buffer, 1, size, pipe);
    }

    int Wait() const
    {
        auto status = pclose(pipe);
        return WEXITSTATUS(status);
    }

    void WorkingDirectory(const fs::path& path) { cwd = path.string(); }
    void EnvironmentVariables(const std::map<std::string_view, std::string_view>& map)
    {
        envs.clear();
        for(const auto& [name, value] : map)
            envs += name + "=" + value + " ";
    }

private:
    std::string command;
    FILE* pipe = nullptr;
    std::string args;
    std::string cwd;
    std::string envs;
};

#endif

Process::Process(const fs::path& cmd, std::string_view args)
    : impl{std::make_unique<ProcessImpl>(cmd.string(), args)}
{
}

Process::~Process() noexcept = default;

Process::Process(Process&&) noexcept = default;
Process& Process::operator=(Process&&) noexcept = default;

Process& Process::WorkingDirectory(const fs::path& cwd)
{
    impl->WorkingDirectory(cwd);
    return *this;
}

Process& Process::EnvironmentVariables(const std::map<std::string_view, std::string_view>& vars)
{
    impl->EnvironmentVariables(vars);
    return *this;
}

const Process& Process::Execute() const
{
    impl->Execute();
    return *this;
}

const Process& Process::Read(std::vector<char>& buffer) const
{
    impl->Read(buffer);
    return *this;
}

const Process& Process::Read(std::string& buffer) const
{
    impl->Read(buffer);
    return *this;
}

const Process& Process::Write(const void* buffer, const std::size_t size) const
{
    impl->Write(buffer, size);
    return *this;
}

int Process::Wait() const { return impl->Wait(); }

} // namespace miopen
