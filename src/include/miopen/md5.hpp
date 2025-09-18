// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef GUARD_MLOPEN_MD5_HPP
#define GUARD_MLOPEN_MD5_HPP

#include <miopen/config.hpp>
#include <string>
#include <vector>

namespace miopen {

MIOPEN_INTERNALS_EXPORT std::string md5(const std::string&);
MIOPEN_INTERNALS_EXPORT std::string md5(const std::vector<char>&);

} // namespace miopen

#endif
