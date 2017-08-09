#ifndef GUARD_MLOPEN_BINARY_CACHE_HPP
#define GUARD_MLOPEN_BINARY_CACHE_HPP

#include <string>

namespace miopen {

std::string LoadBinary(const std::string& name, const std::string& args, bool is_kernel_str);
void SaveBinary(const std::string& binary_path,
                const std::string& name,
                const std::string& args,
                bool is_kernel_str);

} // namespace miopen

#endif
