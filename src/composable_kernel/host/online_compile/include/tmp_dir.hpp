#ifndef GUARD_OLC_TMP_DIR_HPP
#define GUARD_OLC_TMP_DIR_HPP

#include <string>
#include <boost/filesystem/path.hpp>

namespace online_compile {

void SystemCmd(std::string cmd);

struct TmpDir
{
    boost::filesystem::path path;
    TmpDir(std::string prefix);

    TmpDir(TmpDir const&) = delete;
    TmpDir& operator=(TmpDir const&) = delete;

    void Execute(std::string exe, std::string args) const;

    ~TmpDir();
};

} // namespace online_compile

#endif
