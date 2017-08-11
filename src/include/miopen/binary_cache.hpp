#ifndef GUARD_MLOPEN_BINARY_CACHE_HPP
#define GUARD_MLOPEN_BINARY_CACHE_HPP

#include <string>
#include <boost/filesystem/path.hpp>

namespace miopen {

boost::filesystem::path GetCachePath();
std::string LoadBinary(const std::string& device,
                       const std::string& name,
                       const std::string& args,
                       bool is_kernel_str = false);
void SaveBinary(const boost::filesystem::path& binary_path,
                const std::string& device,
                const std::string& name,
                const std::string& args,
                bool is_kernel_str = false);

} // namespace miopen

#endif
