#ifndef GUARD_MLOPEN_BINARY_CACHE_HPP
#define GUARD_MLOPEN_BINARY_CACHE_HPP

#include <string>

namespace miopen {

std::string LoadBinary(const std::string& name, const std::string& args);
void SaveBinary(const std::string& binary_path, const std::string& name, const std::string& args);

} // namespace miopen

#endif
