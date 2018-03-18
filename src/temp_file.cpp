#include <miopen/temp_file.hpp>
#include <miopen/errors.hpp>

#ifndef WIN32
#include <unistd.h>
#else
#include <Windows.h>
#include <fcntl.h>
#include <io.h>
#define _O_EXCL 0x0400
#define O_EXCL _O_EXCL

static int mkstemp(char* tmpl)
{

    static LARGE_INTEGER string_value;
    LARGE_INTEGER random_time;
    int fd;
    int save_errno = errno;

    QueryPerformanceCounter(&random_time);

    string_value.HighPart += random_time.HighPart;
    string_value.LowPart += random_time.LowPart ^ GetCurrentThreadId();

    std::string nm = std::string(tmpl) + std::string("-") + std::to_string(string_value.HighPart) +
                     std::string("-") + std::to_string(string_value.LowPart);

    fd = open(nm.c_str(), O_RDWR | O_CREAT | O_EXCL, _S_IREAD | _S_IWRITE);
    if(fd >= 0)
    {
        errno = save_errno;
        return fd;
    }
    else if(errno != EEXIST)
        return -1;
}
#endif

namespace miopen {
TempFile::TempFile(const std::string& path_template)
#ifdef WIN32
    : _path((GetTempDirectoryPath().path / (path_template)).string())
#else
    : _path((GetTempDirectoryPath().path / (path_template + "-XXXXXX")).string())
#endif

{
    _fd = mkstemp(&_path[0]);
    if(_fd == -1)
    {
        MIOPEN_THROW("Error: TempFile: mkstemp()");
    }
}

TempFile::~TempFile()
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

const TmpDir& TempFile::GetTempDirectoryPath()
{
    static const TmpDir dir("tmp");
    return dir;
}
} // namespace miopen
