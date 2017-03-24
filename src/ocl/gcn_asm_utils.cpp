#include <mlopen/gcn_asm_utils.h>
#include <mlopen/errors.hpp>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <sstream>
#include <cstdio>

#ifndef WIN32
#include <ext/stdio_filebuf.h>
#include <unistd.h>
#include <sys/wait.h>
#endif // !WIN32

static std::string CleanupPath(const char * const p);
static int ExecuteGcnAssembler(const std::string& p, std::vector<std::string>& args, std::istream* in, std::ostream* out);

#ifndef WIN32
class Pipe
{
public:
    Pipe()
        : _read_side_closed(true)
        , _write_side_closed(true)
    {
    }

    Pipe(Pipe&&) = delete;
    Pipe(Pipe&) = delete;
    Pipe& operator =(Pipe&) = delete;
    ~Pipe() { Close(); }
    void CloseRead() { CloseSide(_read_side, _read_side_closed); }
    void CloseWrite() { CloseSide(_write_side, _write_side_closed); }
    int DupRead(int target_fd) { assert(!_read_side_closed); return dup2(_read_side, target_fd); }
    int DupWrite(int target_fd) { assert(!_write_side_closed); return dup2(_write_side, target_fd); }
    int GetReadFd() { return _read_side; }
    int GetWriteFd() { return _write_side; }

    void Close()
    {
        CloseRead();
        CloseWrite();
    }

    void Open()
    {
        if (pipe(_sides)) { MLOPEN_THROW("Error: pipe()"); }
        _read_side_closed = false;
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
        if (closed) { return; }
        if (close(fd)) { std::fprintf(stderr, "Error closing pipe"); }
        closed = true;
    }
};
#endif // !WIN32

std::string GetGcnAssemblerPath()
{
    const auto asm_path_env_p = std::getenv("MLOPEN_EXPERIMENTAL_GCN_ASM_PATH");
    if (asm_path_env_p) {
        return CleanupPath(asm_path_env_p);
    }
    static const std::string defaultAsmPath = "/opt/rocm/opencl/bin/x86_64/clang";
    return defaultAsmPath;
}

bool ValidateGcnAssembler()
{
#ifndef WIN32
    const auto path = GetGcnAssemblerPath();
    if (path.empty()) { return false; }
    if (!std::ifstream(path).good()) { return false; }
    
    std::vector<std::string> args({"--version"});
    std::stringstream clang_stdout;
    std::string clang_result_line;
    auto clang_rc = ExecuteGcnAssembler(path, args, nullptr, &clang_stdout);

    if (clang_rc != 0) { return false; }

    std::getline(clang_stdout, clang_result_line);
    if (clang_result_line.find("clang") != std::string::npos)

    while (!clang_stdout.eof()) {
        std::getline(clang_stdout, clang_result_line);

        if (clang_result_line.find("Target: ") != std::string::npos) {
            return clang_result_line.find("amdgcn") != std::string::npos;
        }
    }
#endif // !WIN32
    return false;
}

static
int ExecuteGcnAssembler(const std::string& p, std::vector<std::string>& args, std::istream* in, std::ostream* out)
{
#ifndef WIN32
    Pipe clang_stdin;
    Pipe clang_stdout;

    const auto redirect_stdin = (in != nullptr);
    const auto redirect_stdout = (out != nullptr);

    if (redirect_stdin) { clang_stdin.Open(); }
    if (redirect_stdout) { clang_stdout.Open(); }

    int wstatus;
    pid_t pid = fork();
    if (pid == 0) {
        std::string path(p); // to remove constness 
        std::vector<char*> c_args;
        c_args.push_back(&path[0]);
        for (auto& arg : args) {
            c_args.push_back(&arg[0]);
        }
        c_args.push_back(nullptr);

        if (redirect_stdin) {
            if (clang_stdin.DupRead(STDIN_FILENO) == -1) { std::exit(EXIT_FAILURE); }
            clang_stdin.Close();
        }

        if (redirect_stdout) {
            if (clang_stdout.DupWrite(STDOUT_FILENO) == -1) { std::exit(EXIT_FAILURE); }
            clang_stdout.Close();
        }

        clang_stdout.Close();

        execv(path.c_str(), c_args.data());
        std::exit(EXIT_FAILURE);
    }
    else {
        if (pid == -1) { MLOPEN_THROW("Error X-AMDGCN-ASM: fork()"); }

        if (redirect_stdin)
        {
            clang_stdin.CloseRead();
            __gnu_cxx::stdio_filebuf<char> clang_stdin_buffer(clang_stdin.GetWriteFd(), std::ios::out);
            std::ostream clang_stdin_stream(&clang_stdin_buffer);
            clang_stdin_stream << in->rdbuf();
            clang_stdin.CloseWrite();
        }

        if (redirect_stdout)
        {
            clang_stdout.CloseWrite();
            __gnu_cxx::stdio_filebuf<char> clang_stdout_buffer(clang_stdout.GetReadFd(), std::ios::in);
            std::istream clang_stdin_stream(&clang_stdout_buffer);
            *out << clang_stdin_stream.rdbuf();
            clang_stdout.CloseRead();
        }

        if (waitpid(pid, &wstatus, 0) != pid) { MLOPEN_THROW("Error: X-AMDGCN-ASM: waitpid()"); }
    }

    if (WIFEXITED(wstatus)) {
        const int exit_status = WEXITSTATUS(wstatus);
        return exit_status;
    }
    else {
        MLOPEN_THROW("Error: X-AMDGCN-ASM: clang terminated abnormally");
    }
#else
    (void)p;
    (void)args;
    (void)in;
    (void)out;
    return -1;
#endif // !WIN32
}

int ExecuteGcnAssembler(std::vector<std::string>& args, std::istream* clang_stdin_content, std::ostream* clang_stdout_content)
{
    auto path = GetGcnAssemblerPath();
    return ExecuteGcnAssembler(path, args, clang_stdin_content, clang_stdout_content);
}

static
std::string CleanupPath(const char * const p)
{
    std::string path(p);
    static const char bad[] = "!#$*;<>?@\\^`{|}";
    for (char * c = &path[0]; c < (&path[0] + path.length()); ++c) {
        if (std::iscntrl(*c)) {
            *c = '_';
            continue;
        }
        for (const char * b = &bad[0]; b < (&bad[0] + sizeof(bad) - 1); ++b) {
            if (*b == *c) {
                *c = '_';
                break;
            }
        }
    }
    return path;
}
