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
#include <string_view>

#ifdef _WIN32

#define WIN32_MEAN_AND_LEAN
#include <Windows.h>

class Process
{
public:
    Process(std::string_view title, std::string_view cmd, std::string_view cwd = "")
    {
        STARTUPINFOA info;
        ZeroMemory(&info, sizeof(STARTUPINFO));
        info.cb = sizeof(STARTUPINFO);

        // Refer to
        // https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessa
        constexpr std::size_t BUFFER_CAPACITY = 32767;

        TCHAR buffer[BUFFER_CAPACITY];
        std::strncpy(buffer, cmd.data(), BUFFER_CAPACITY);

        if(CreateProcess(title.data(),
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

    ~Process()
    {
        CloseHandle(processInfo.hProcess);
        CloseHandle(processInfo.hThread);
    }

    int Wait()
    {
        WaitForSingleObject(processInfo.hProcess, INFINITE);

        DWORD status;
        const auto getExitCodeStatus = GetExitCodeProcess(processInfo.hProcess, &status);

        if(getExitCodeStatus == 0)
            MIOPEN_THROW("GetExitCodeProcess error: " + std::to_string(GetLastError()));

        return status;
    }

private:
    PROCESS_INFORMATION processInfo{};
};

#else

class Process
{
public:
    Process(std::string_view, std::string_view cmd, std::string_view cwd = "")
    {
        std::string command;
        if(not cwd.empty())
        {
            command = "cd " + std::string{cwd} + ";";
        }
        command += cmd;
        pipe = popen(command.c_str(), "w");
        if(pipe == nullptr)
            MIOPEN_THROW("Error: popen()");
    }

    int Wait()
    {
        auto status = pclose(pipe);
        return WEXITSTATUS(status);
    }

private:
    FILE* pipe = nullptr;
};

#endif
