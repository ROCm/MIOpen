#ifndef MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP
#define MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP

#include <string_view>
#include <boost/filesystem/path.hpp>

namespace miopen {

struct TmpDir
{
    boost::filesystem::path path;
    explicit TmpDir(std::string_view prefix = "");

    TmpDir(TmpDir const&) = delete;
    TmpDir& operator=(TmpDir const&) = delete;

    TmpDir(TmpDir&& other) noexcept { (*this) = std::move(other); }
    TmpDir& operator=(TmpDir&& other) noexcept;

    int Execute(const boost::filesystem::path& exec, std::string_view args) const;

    ~TmpDir();
};

} // namespace miopen

#endif
