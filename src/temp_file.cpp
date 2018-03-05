#include <miopen/temp_file.hpp>
#include <miopen/errors.hpp>

#include <unistd.h>

namespace miopen {
TempFile::TempFile(const std::string& path_template)
    : _path((GetTempDirectoryPath().path / (path_template + "-XXXXXX")).string())
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
