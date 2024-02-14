#ifndef MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP
#define MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP

#include <string_view>
#include <miopen/filesystem.hpp>

namespace miopen {

struct TmpDir
{
    fs::path path;
    TmpDir(std::string prefix);

    TmpDir(TmpDir const&) = delete;
    TmpDir& operator=(TmpDir const&) = delete;

    TmpDir(TmpDir&& other) noexcept { (*this) = std::move(other); }
    TmpDir& operator=(TmpDir&& other) noexcept;

    void Execute(std::string_view exe, std::string_view args) const;

    ~TmpDir();
};

} // namespace miopen

#endif
