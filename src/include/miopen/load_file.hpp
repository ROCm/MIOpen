#ifndef MIOPEN_GUARD_MLOPEN_LOAD_FILE_HPP
#define MIOPEN_GUARD_MLOPEN_LOAD_FILE_HPP

#include <miopen/filesystem.hpp>
#include <string>

namespace miopen {

std::string LoadFile(const std::string& s);
std::string LoadFile(const fs::path& p);

} // namespace miopen

#endif
