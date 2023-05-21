#ifndef MIOPEN_GUARD_MLOPEN_LOAD_FILE_HPP
#define MIOPEN_GUARD_MLOPEN_LOAD_FILE_HPP

#include <boost/filesystem/path.hpp>
#include <string>
#include <vector>

namespace miopen {

std::string LoadFile(const std::string& path);
inline std::string LoadFile(const boost::filesystem::path& path) { return LoadFile(path.string()); }
std::vector<char> LoadFileAsVector(const std::string& path);
inline std::vector<char> LoadFileAsVector(const boost::filesystem::path& path)
{
    return LoadFileAsVector(path.string());
}

} // namespace miopen

#endif
