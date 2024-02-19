#ifndef MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP
#define MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP

#include <string_view>
#include <miopen/filesystem.hpp>

namespace miopen {

struct TmpDir
{
    fs::path path;
    explicit TmpDir(std::string_view prefix = "");

    TmpDir(TmpDir&&) = default;
    TmpDir& operator = (TmpDir&&) = default;

    fs::path operator / (std::string_view other) const { return path / other; }

    operator const fs::path& () const { return path; }
    operator std::string () const { return path.string(); }

    int Execute(std::string_view cmd, std::string_view args) const;
    int Execute(const fs::path& exec, std::string_view args) const
    {
        return Execute(std::string_view{exec.string()}, args);
    }

    ~TmpDir();
};

} // namespace miopen

#endif
