#ifndef MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP
#define MIOPEN_GUARD_MLOPEN_TMP_DIR_HPP

#include <string>
#include <boost/filesystem/path.hpp>

namespace miopen {

void SystemCmd(std::string cmd);

struct TmpDir
{
    boost::filesystem::path path;
    TmpDir(std::string prefix);

    TmpDir(TmpDir const&) = delete;
    TmpDir& operator=(TmpDir const&) = delete;

    TmpDir(TmpDir&& other) noexcept { (*this) = std::move(other); }
    TmpDir& operator=(TmpDir&& other) noexcept;

    void Execute(std::string exe, std::string args) const;

    ~TmpDir();
};

} // namespace miopen

#endif
