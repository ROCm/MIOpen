#include <miopen/temp_file.hpp>
#include <miopen/env.hpp>
#include <miopen/errors.hpp>

#include <unistd.h>

struct tmp_dir_env
{
    static const char* value() { return "TMPDIR"; }
};

namespace miopen {
    TempFile::TempFile(const std::string& path_template)
        : _path(GetTempDirectoryPath() + "/" + path_template + "-XXXXXX")
    {
        _fd = mkstemp(&_path[0]);
        if (_fd == -1)
        {
            MIOPEN_THROW("Error: TempFile: mkstemp()");
        }
    }

    TempFile::~TempFile()
    {
        const int remove_rc = std::remove(_path.c_str());
        const int close_rc = close(_fd);
        if (remove_rc != 0 || close_rc != 0)
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

    std::string TempFile::GetTempDirectoryPath()
    {
        const auto path = miopen::GetStringEnv(tmp_dir_env{});
        if (path != nullptr)
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
} // namespace miopen
