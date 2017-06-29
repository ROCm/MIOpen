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
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <miopen/config.h>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <sstream>

#ifdef __linux__
#include <ext/stdio_filebuf.h>
#include <paths.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif // __linux__

MIOPEN_DECLARE_ENV_VAR(MIOPEN_EXPERIMENTAL_GCN_ASM_PATH)

struct tmp_dir_env
{
    static const char* value() { return "TMPDIR"; }
};

#ifdef __linux__
class TempFile
{
    public:
    TempFile(const std::string& path_template)
        : _path(GetTempDirectoryPath() + "/" + path_template + "-XXXXXX")
    {
        _fd = mkstemp(&_path[0]);
        if(_fd == -1)
        {
            MIOPEN_THROW("Error: TempFile: mkstemp()");
        }
    }

    ~TempFile()
    {
        const int remove_rc = std::remove(_path.c_str());
        const int close_rc  = close(_fd);
        if(remove_rc != 0 || close_rc != 0)
        {
#ifndef NDEBUG // Be quiet in release versions.
            std::fprintf(stderr,
                         "Error: TempFile: On removal of '%s', remove_rc = %d, close_rc = %d.\n",
                         _path.c_str(),
                         remove_rc,
                         close_rc);
#endif
        }
    }

    inline operator const std::string&() { return _path; }

    private:
    std::string _path;
    int _fd;

    static const std::string GetTempDirectoryPath()
    {
        const auto path = miopen::GetStringEnv(tmp_dir_env{});
        if(path != nullptr)
        {
            return path;
        }
#if defined(P_tmpdir)
        return P_tmpdir; // a string literal, if defined.
#elif defined(_PATH_TMP)
        return _PATH_TMP; // a string literal, if defined.
#else
        return "/tmp";
#endif
    }
};
#endif

static std::string CleanupPath(const char* p);
static int ExecuteGcnAssembler(const std::string& p,
                               std::vector<std::string>& args,
                               std::istream* in,
                               std::ostream* out);

#ifdef __linux__
class Pipe
{
    public:
    Pipe() : _read_side_closed(true), _write_side_closed(true) {}

    Pipe(Pipe&&)  = delete;
    Pipe(Pipe&)   = delete;
    Pipe& operator=(Pipe&) = delete;
    ~Pipe() { Close(); }
    void CloseRead() { CloseSide(_read_side, _read_side_closed); }
    void CloseWrite() { CloseSide(_write_side, _write_side_closed); }
    int DupRead(int target_fd)
    {
        assert(!_read_side_closed);
        return dup2(_read_side, target_fd);
    }
    int DupWrite(int target_fd)
    {
        assert(!_write_side_closed);
        return dup2(_write_side, target_fd);
    }
    int GetReadFd() { return _read_side; }
    int GetWriteFd() { return _write_side; }

    void Close()
    {
        CloseRead();
        CloseWrite();
    }

    void Open()
    {
        if(pipe(_sides))
        {
            MIOPEN_THROW("Error: pipe()");
        }
        _read_side_closed  = false;
        _write_side_closed = false;
    }

    private:
    union
    {
        int _sides[2];
        struct
        {
            int _read_side;
            int _write_side;
        };
    };

    bool _read_side_closed;
    bool _write_side_closed;

    static void CloseSide(int fd, bool& closed)
    {
        if(closed)
        {
            return;
        }
        if(close(fd))
        {
            std::fprintf(stderr, "Error closing pipe");
        }
        closed = true;
    }
};
#endif // __linux__

std::string GetGcnAssemblerPathImpl()
{
    const auto asm_path_env_p = miopen::GetStringEnv(MIOPEN_EXPERIMENTAL_GCN_ASM_PATH{});
    if(asm_path_env_p)
    {
        return CleanupPath(asm_path_env_p);
    }
#ifdef MIOPEN_AMDGCN_ASSEMBLER // string literal generated by CMake
    return CleanupPath(MIOPEN_AMDGCN_ASSEMBLER);
#else
    return "";
#endif
}

std::string GetGcnAssemblerPath()
{
    static const auto result = GetGcnAssemblerPathImpl();
    return result;
}

bool ValidateGcnAssemblerImpl()
{
#ifdef __linux__
    const auto path = GetGcnAssemblerPath();
    if(path.empty())
    {
        return false;
    }
    if(!std::ifstream(path).good())
    {
        return false;
    }

    std::vector<std::string> args({"--version"});
    std::stringstream clang_stdout;
    std::string clang_result_line;
    auto clang_rc = ExecuteGcnAssembler(path, args, nullptr, &clang_stdout);

    if(clang_rc != 0)
    {
        return false;
    }

    std::getline(clang_stdout, clang_result_line);
    if(clang_result_line.find("clang") != std::string::npos)
    {
        while(!clang_stdout.eof())
        {
            std::getline(clang_stdout, clang_result_line);
            if(clang_result_line.find("Target: ") != std::string::npos)
            {
                return clang_result_line.find("amdgcn") != std::string::npos;
            }
        }
    }
#endif // __linux__
    return false;
}

bool ValidateGcnAssembler()
{
    static bool result = ValidateGcnAssemblerImpl();
    return result;
}

int ExecuteGcnAssembler(const std::string& p,
                        std::vector<std::string>& args,
                        std::istream* in,
                        std::ostream* out)
{
#ifdef __linux__
    Pipe clang_stdin;
    Pipe clang_stdout;

    const auto redirect_stdin  = (in != nullptr);
    const auto redirect_stdout = (out != nullptr);

    if(redirect_stdin)
    {
        clang_stdin.Open();
    }
    if(redirect_stdout)
    {
        clang_stdout.Open();
    }

    int wstatus;
    pid_t pid = fork();
    if(pid == 0)
    {
        std::string path(p); // to remove constness
        std::vector<char*> c_args;
        c_args.push_back(&path[0]);
        for(auto& arg : args)
        {
            c_args.push_back(&arg[0]);
        }
        c_args.push_back(nullptr);

        if(redirect_stdin)
        {
            if(clang_stdin.DupRead(STDIN_FILENO) == -1)
            {
                std::exit(EXIT_FAILURE);
            }
            clang_stdin.Close();
        }

        if(redirect_stdout)
        {
            if(clang_stdout.DupWrite(STDOUT_FILENO) == -1)
            {
                std::exit(EXIT_FAILURE);
            }
            clang_stdout.Close();
        }

        clang_stdout.Close();

        execv(path.c_str(), c_args.data());
        std::exit(EXIT_FAILURE);
    }
    else
    {
        if(pid == -1)
        {
            MIOPEN_THROW("Error X-AMDGCN-ASM: fork()");
        }

        if(redirect_stdin)
        {
            clang_stdin.CloseRead();
            __gnu_cxx::stdio_filebuf<char> clang_stdin_buffer(clang_stdin.GetWriteFd(),
                                                              std::ios::out);
            std::ostream clang_stdin_stream(&clang_stdin_buffer);
            clang_stdin_stream << in->rdbuf();
            clang_stdin.CloseWrite();
        }

        if(redirect_stdout)
        {
            clang_stdout.CloseWrite();
            __gnu_cxx::stdio_filebuf<char> clang_stdout_buffer(clang_stdout.GetReadFd(),
                                                               std::ios::in);
            std::istream clang_stdin_stream(&clang_stdout_buffer);
            *out << clang_stdin_stream.rdbuf();
            clang_stdout.CloseRead();
        }

        if(waitpid(pid, &wstatus, 0) != pid)
        {
            MIOPEN_THROW("Error: X-AMDGCN-ASM: waitpid()");
        }
    }

    if(WIFEXITED(wstatus))
    {
        const int exit_status = WEXITSTATUS(wstatus);
        return exit_status;
    }
    else
    {
        MIOPEN_THROW("Error: X-AMDGCN-ASM: clang terminated abnormally");
    }
#else
    (void)p;
    (void)args;
    (void)in;
    (void)out;
    return -1;
#endif // __linux__
}

int ExecuteGcnAssembler(std::vector<std::string>& args,
                        std::istream* clang_stdin_content,
                        std::ostream* clang_stdout_content)
{
    auto path = GetGcnAssemblerPath();
    return ExecuteGcnAssembler(path, args, clang_stdin_content, clang_stdout_content);
}

static std::string CleanupPath(const char* p)
{
    std::string path(p);
    static const char bad[] = "!#$*;<>?@\\^`{|}";
    for(char* c = &path[0]; c < (&path[0] + path.length()); ++c)
    {
        if(std::iscntrl(*c))
        {
            *c = '_';
            continue;
        }
        for(const char* b = &bad[0]; b < (&bad[0] + sizeof(bad) - 1); ++b)
        {
            if(*b == *c)
            {
                *c = '_';
                break;
            }
        }
    }
    return path;
}

/*
 * Temporary function which emulates online assembly feature of OpenCL-on-ROCm being developed.
 * Not intended to be used in production code, so error handling is very straghtforward,
 * just catch whatever possible and throw an exception.
 */
void AmdgcnAssemble(std::string& source, const std::string& params)
{
#ifdef __linux__
    TempFile outfile("amdgcn-asm-out-XXXXXX");

    std::vector<std::string> args({
        "-x", "assembler", "-target", "amdgcn--amdhsa",
    });

    {
        std::istringstream iss(params);
        std::string param;
        while(iss >> param)
        {
            args.push_back(param);
        };
    }
    args.push_back("-");
    args.push_back("-o");
    args.push_back(outfile);

    std::istringstream clang_stdin(source);
    const auto clang_rc = ExecuteGcnAssembler(args, &clang_stdin, nullptr);
    if(clang_rc != 0)
        MIOPEN_THROW("Assembly error(" + std::to_string(clang_rc) + ")");

    std::ifstream file(outfile, std::ios::binary | std::ios::ate);
    bool outfile_read_failed = false;
    do
    {
        const auto size = file.tellg();
        if(size == -1)
        {
            outfile_read_failed = true;
            break;
        }
        source.resize(size, '\0');
        file.seekg(std::ios::beg);
        if(file.fail())
        {
            outfile_read_failed = true;
            break;
        }
        if(file.rdbuf()->sgetn(&source[0], size) != size)
        {
            outfile_read_failed = true;
            break;
        }
    } while(false);
    file.close();
    if(outfile_read_failed)
    {
        MIOPEN_THROW("Error: X-AMDGCN-ASM: outfile_read_failed");
    }
#else
    (void)source; // -warning
    (void)params; // -warning
    MIOPEN_THROW("Error: X-AMDGCN-ASM: online assembly under Windows is not supported");
#endif // __linux__
}
