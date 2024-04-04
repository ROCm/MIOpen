#ifndef GUARD_MLOPEN_MD5_HPP
#define GUARD_MLOPEN_MD5_HPP

#include <string>
#include <vector>

namespace miopen {

std::string md5(const std::string&);
std::string md5(const std::vector<char>&);

} // namespace miopen

#endif
