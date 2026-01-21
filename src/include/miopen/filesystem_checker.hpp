// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <miopen/config.hpp>
#include <miopen/filesystem.hpp>

namespace miopen {

/// Abstract interface for filesystem operations that need to be mockable for testing
class IFilesystemChecker
{
public:
    virtual ~IFilesystemChecker() = default;

    /// Check if the given path is on a networked filesystem
    /// @param path The filesystem path to check
    /// @return true if the path is on a networked filesystem, false otherwise
    virtual bool IsNetworkedFilesystem(const fs::path& path) const = 0;
};

/// Default implementation that uses the actual system calls
class FilesystemChecker : public IFilesystemChecker
{
public:
    bool IsNetworkedFilesystem(const fs::path& path) const override;
};

/// Get the global filesystem checker instance
/// @return Reference to the current filesystem checker (default or test override)
MIOPEN_INTERNALS_EXPORT const IFilesystemChecker& GetFilesystemChecker();

/// Set a custom filesystem checker (for testing purposes only)
/// @param checker Pointer to custom checker, or nullptr to restore default
MIOPEN_INTERNALS_EXPORT void SetFilesystemChecker(IFilesystemChecker* checker);

} // namespace miopen
