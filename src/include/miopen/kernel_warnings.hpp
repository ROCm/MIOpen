#ifndef GUARD_MLOPEN_KERNEL_WARNINGS_HPP
#define GUARD_MLOPEN_KERNEL_WARNINGS_HPP

#include <vector>
#include <string>

namespace miopen {

std::vector<std::string> KernelWarnings();
const std::string& KernelWarningsString();

} // namespace miopen


#endif



