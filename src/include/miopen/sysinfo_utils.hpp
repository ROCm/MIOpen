// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <miopen/config.hpp>
#include <string>

namespace miopen {
namespace sysinfo {

/// Retrieves the system hostname for logging and identification purposes
MIOPEN_INTERNALS_EXPORT std::string GetSystemHostname();

} // namespace sysinfo
} // namespace miopen
