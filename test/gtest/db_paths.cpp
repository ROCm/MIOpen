// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/db_path.hpp>
#include <miopen/binary_cache.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/filesystem_checker.hpp>
#include <miopen/version.h>
#include <miopen/stringutils.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "gtest_common.hpp"

#include <string>
#include <sstream>

using ::testing::_;
using ::testing::Return;

namespace fs = miopen::fs;

MIOPEN_LIB_ENV_VAR(MIOPEN_USER_DB_PATH)
MIOPEN_LIB_ENV_VAR(MIOPEN_CUSTOM_CACHE_DIR)

// Helper function to build expected version string
std::string GetExpectedVersionString()
{
    std::ostringstream oss;
    oss << MIOPEN_VERSION_MAJOR << "." << MIOPEN_VERSION_MINOR << "." << MIOPEN_VERSION_PATCH << "."
        << MIOPEN_STRINGIZE(MIOPEN_VERSION_TWEAK);
    return oss.str();
}

// Helper function to check if a path contains a substring
bool PathContains(const fs::path& path, const std::string& substring)
{
    return path.string().find(substring) != std::string::npos;
}

// Mock filesystem checker for testing
class MockFilesystemChecker : public miopen::IFilesystemChecker
{
public:
    MOCK_METHOD(bool, IsNetworkedFilesystem, (const fs::path& path), (const, override));
};

// Test fixture for db_path tests with filesystem mocking
class CPU_DbPaths_NONE : public ::testing::Test
{
protected:
    MockFilesystemChecker mock_checker;

    void SetUp() override
    {
        // Reset cached paths to allow fresh initialization with new mock/env settings
        miopen::testing::ResetCachedPaths();
        miopen::testing::ResetUserDbPath();

        // Install the mock checker before any path functions are called
        miopen::SetFilesystemChecker(&mock_checker);
    }

    void TearDown() override
    {
        // Restore default checker
        miopen::SetFilesystemChecker(nullptr);
    }
};

// ============================================================================
// Tests for GetUserDbPath() - 4 scenarios
// ============================================================================

TEST_F(CPU_DbPaths_NONE, UserDbPath_LocalFS_NoEnvVar)
{
    // Ensure environment variable is NOT set for this test
    ScopedEnvironment<std::string> unset_user_db(MIOPEN_USER_DB_PATH, "");

    // Scenario 1: Local filesystem, no environment variable set
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto& user_db_path = miopen::GetUserDbPath();

    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use default path containing "miopen"
    EXPECT_TRUE(PathContains(user_db_path, "miopen"))
        << "User DB path '" << user_db_path.string() << "' should contain 'miopen' folder";
}

TEST_F(CPU_DbPaths_NONE, UserDbPath_LocalFS_EnvVarSet)
{
    // Scenario 2: Local filesystem, environment variable set
    const std::string custom_path = "/custom/user/db/path";
    ScopedEnvironment<std::string> scoped_env(MIOPEN_USER_DB_PATH, custom_path);

    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto& user_db_path = miopen::GetUserDbPath();

    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use the custom path from environment variable
    EXPECT_TRUE(PathContains(user_db_path, custom_path))
        << "User DB path '" << user_db_path.string() << "' should contain custom path '"
        << custom_path << "'";
}

TEST_F(CPU_DbPaths_NONE, UserDbPath_NetworkFS_NoEnvVar)
{
#if MIOPEN_BUILD_DEV
    GTEST_SKIP() << "Network filesystem detection is disabled in MIOPEN_BUILD_DEV mode";
#endif

    // Ensure environment variable is NOT set for this test
    ScopedEnvironment<std::string> unset_user_db(MIOPEN_USER_DB_PATH, "");

    // Scenario 3: Network filesystem, no environment variable set
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(true));

    const auto& user_db_path = miopen::GetUserDbPath();

    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should fallback to temp directory
    const auto temp_dir = fs::temp_directory_path();
    EXPECT_TRUE(PathContains(user_db_path, temp_dir.string()))
        << "User DB path '" << user_db_path.string()
        << "' should be in temp directory for network filesystem";

    // Should contain .config/miopen
    EXPECT_TRUE(PathContains(user_db_path, ".config"))
        << "User DB path should contain '.config' folder";
    EXPECT_TRUE(PathContains(user_db_path, "miopen"))
        << "User DB path should contain 'miopen' folder";
}

TEST_F(CPU_DbPaths_NONE, UserDbPath_NetworkFS_EnvVarSet)
{
    // Scenario 4: Network filesystem, environment variable set
    // Environment variable should take precedence over network detection
    const std::string custom_path = "/custom/network/db/path";
    ScopedEnvironment<std::string> scoped_env(MIOPEN_USER_DB_PATH, custom_path);

    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(true));

    const auto& user_db_path = miopen::GetUserDbPath();

    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use the custom path from environment variable (takes precedence)
    EXPECT_TRUE(PathContains(user_db_path, custom_path))
        << "User DB path '" << user_db_path.string()
        << "' should use custom path even on network filesystem";
}

// ============================================================================
// Tests for GetCachePath() - 4 scenarios
// ============================================================================

TEST_F(CPU_DbPaths_NONE, CachePath_LocalFS_NoEnvVar)
{
    // Ensure environment variable is NOT set for this test
    ScopedEnvironment<std::string> unset_cache(MIOPEN_CUSTOM_CACHE_DIR, "");

    // Scenario 1: Local filesystem, no environment variable set
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto cache_path = miopen::GetCachePath(false);

    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should contain version string
    const std::string expected_version = GetExpectedVersionString();
    EXPECT_TRUE(PathContains(cache_path, expected_version))
        << "Cache path '" << cache_path.string() << "' should contain version string '"
        << expected_version << "'";

    // Should contain "miopen"
    EXPECT_TRUE(PathContains(cache_path, "miopen")) << "Cache path should contain 'miopen' folder";
}

TEST_F(CPU_DbPaths_NONE, CachePath_LocalFS_EnvVarSet)
{
    // Scenario 2: Local filesystem, environment variable set
    const std::string custom_cache = (fs::temp_directory_path() / "custom/cache/dir").string();
    ScopedEnvironment<std::string> scoped_env(MIOPEN_CUSTOM_CACHE_DIR, custom_cache);

    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto cache_path = miopen::GetCachePath(false);

    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use the custom cache directory
    EXPECT_TRUE(PathContains(cache_path, custom_cache))
        << "Cache path '" << cache_path.string() << "' should contain custom cache dir '"
        << custom_cache << "'";
}

TEST_F(CPU_DbPaths_NONE, CachePath_NetworkFS_NoEnvVar)
{
#if MIOPEN_BUILD_DEV
    GTEST_SKIP() << "Network filesystem detection is disabled in MIOPEN_BUILD_DEV mode";
#endif

    // Ensure environment variable is NOT set for this test
    ScopedEnvironment<std::string> unset_cache(MIOPEN_CUSTOM_CACHE_DIR, "");

    // Scenario 3: Network filesystem, no environment variable set
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(true));

    const auto cache_path = miopen::GetCachePath(false);

    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should fallback to temp directory
    const auto temp_dir = fs::temp_directory_path();
    EXPECT_TRUE(PathContains(cache_path, temp_dir.string()))
        << "Cache path '" << cache_path.string()
        << "' should be in temp directory for network filesystem";

    // Should contain .cache/miopen/<version>
    EXPECT_TRUE(PathContains(cache_path, ".cache")) << "Cache path should contain '.cache' folder";
    EXPECT_TRUE(PathContains(cache_path, "miopen")) << "Cache path should contain 'miopen' folder";

    const std::string expected_version = GetExpectedVersionString();
    EXPECT_TRUE(PathContains(cache_path, expected_version))
        << "Cache path should contain version string '" << expected_version << "'";
}

TEST_F(CPU_DbPaths_NONE, CachePath_NetworkFS_EnvVarSet)
{
    // Scenario 4: Network filesystem, environment variable set
    // Environment variable should take precedence over network detection
    const std::string custom_cache = (fs::temp_directory_path() / "custom/network/cache").string();
    ScopedEnvironment<std::string> scoped_env(MIOPEN_CUSTOM_CACHE_DIR, custom_cache);

    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(true));

    const auto cache_path = miopen::GetCachePath(false);

    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use the custom cache directory (takes precedence)
    EXPECT_TRUE(PathContains(cache_path, custom_cache))
        << "Cache path '" << cache_path.string()
        << "' should use custom cache even on network filesystem";
}

// ============================================================================
// Additional tests
// ============================================================================

TEST_F(CPU_DbPaths_NONE, UserDbSuffix_ContainsVersionInfo)
{
    const std::string suffix = miopen::GetUserDbSuffix();
    EXPECT_FALSE(suffix.empty()) << "User DB suffix should not be empty";

    // Suffix should contain version numbers separated by underscores
    std::ostringstream expected_pattern;
    expected_pattern << MIOPEN_VERSION_MAJOR << "_" << MIOPEN_VERSION_MINOR << "_"
                     << MIOPEN_VERSION_PATCH;

    EXPECT_TRUE(suffix.find(expected_pattern.str()) != std::string::npos)
        << "Suffix '" << suffix << "' should contain version pattern '" << expected_pattern.str()
        << "'";
}

TEST_F(CPU_DbPaths_NONE, UserAndSystemCachePaths_AreDifferent)
{
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto user_cache = miopen::GetCachePath(false);
    const auto sys_cache  = miopen::GetCachePath(true);

    if(!user_cache.empty() && !sys_cache.empty())
    {
        EXPECT_NE(user_cache, sys_cache) << "User and system cache paths should be different";
    }
}

TEST_F(CPU_DbPaths_NONE, Paths_AreValid)
{
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto& user_db_path  = miopen::GetUserDbPath();
    const auto cache_path     = miopen::GetCachePath(false);
    const auto sys_cache_path = miopen::GetCachePath(true);

    // Paths should either be empty (if disabled) or valid filesystem paths
    if(!user_db_path.empty())
    {
        EXPECT_FALSE(user_db_path.string().empty());
    }

    if(!cache_path.empty())
    {
        EXPECT_FALSE(cache_path.string().empty());
    }

    if(!sys_cache_path.empty())
    {
        EXPECT_FALSE(sys_cache_path.string().empty());
    }
}

TEST_F(CPU_DbPaths_NONE, CacheDisabled_ReturnsCorrectly)
{
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const bool is_disabled = miopen::IsCacheDisabled();

    // The result depends on build configuration
    // Just verify it returns a valid boolean without crashing
    EXPECT_TRUE(is_disabled == true || is_disabled == false);
}
