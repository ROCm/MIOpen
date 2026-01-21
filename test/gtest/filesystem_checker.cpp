// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/filesystem_checker.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::_;
using ::testing::Return;

// Mock implementation of IFilesystemChecker for testing
class MockFilesystemChecker : public miopen::IFilesystemChecker
{
public:
    MOCK_METHOD(bool, IsNetworkedFilesystem, (const miopen::fs::path& path), (const, override));
};

// Helper to automatically restore the default checker after tests
class FilesystemCheckerGuard
{
public:
    FilesystemCheckerGuard(miopen::IFilesystemChecker* checker)
    {
        miopen::SetFilesystemChecker(checker);
    }

    ~FilesystemCheckerGuard() { miopen::SetFilesystemChecker(nullptr); }
};

TEST(CPU_FilesystemChecker_NONE, DefaultCheckerWorks)
{
    // The default checker should be available
    auto& checker = miopen::GetFilesystemChecker();

    // This will use the real implementation
    // We can't predict the result, but it shouldn't crash
    miopen::fs::path test_path = "/tmp";
    auto result                = checker.IsNetworkedFilesystem(test_path);

    // Verify at compile-time that it returns a boolean type
    static_assert(std::is_same<decltype(result), bool>::value,
                  "IsNetworkedFilesystem must return bool");
}

TEST(CPU_FilesystemChecker_NONE, MockCheckerCanBeInjected)
{
    MockFilesystemChecker mock;
    miopen::fs::path test_path = "/some/test/path";

    // Set expectation: IsNetworkedFilesystem will be called once with test_path and return true
    EXPECT_CALL(mock, IsNetworkedFilesystem(test_path)).WillOnce(Return(true));

    FilesystemCheckerGuard guard(&mock);

    auto& checker = miopen::GetFilesystemChecker();
    bool result   = checker.IsNetworkedFilesystem(test_path);

    // Verify the mock returned the expected value
    EXPECT_TRUE(result);
}

TEST(CPU_FilesystemChecker_NONE, MockCheckerReturnsNonNetworked)
{
    MockFilesystemChecker mock;
    miopen::fs::path test_path = "/another/test/path";

    // Set expectation: IsNetworkedFilesystem will be called once with test_path and return false
    EXPECT_CALL(mock, IsNetworkedFilesystem(test_path)).WillOnce(Return(false));

    FilesystemCheckerGuard guard(&mock);

    auto& checker = miopen::GetFilesystemChecker();
    bool result   = checker.IsNetworkedFilesystem(test_path);

    // Verify the mock returned the expected value
    EXPECT_FALSE(result);
}

TEST(CPU_FilesystemChecker_NONE, DefaultCheckerRestoredAfterTest)
{
    {
        MockFilesystemChecker mock;

        // Set expectation: IsNetworkedFilesystem will be called once and return true
        EXPECT_CALL(mock, IsNetworkedFilesystem(miopen::fs::path("/test"))).WillOnce(Return(true));

        FilesystemCheckerGuard guard(&mock);

        // Inside this scope, mock is active
        EXPECT_TRUE(miopen::GetFilesystemChecker().IsNetworkedFilesystem("/test"));
    }

    // After guard goes out of scope, default checker should be restored
    // We can't predict the result, but it should work without crashing
    miopen::fs::path test_path = "/tmp";
    auto result                = miopen::GetFilesystemChecker().IsNetworkedFilesystem(test_path);
    static_assert(std::is_same<decltype(result), bool>::value,
                  "IsNetworkedFilesystem must return bool");
}

TEST(CPU_FilesystemChecker_NONE, MultiplePathsCanBeChecked)
{
    MockFilesystemChecker mock;

    // Set expectations: IsNetworkedFilesystem will be called three times with specific paths
    EXPECT_CALL(mock, IsNetworkedFilesystem(miopen::fs::path("/path1"))).WillOnce(Return(true));
    EXPECT_CALL(mock, IsNetworkedFilesystem(miopen::fs::path("/path2"))).WillOnce(Return(false));
    EXPECT_CALL(mock, IsNetworkedFilesystem(miopen::fs::path("/path3"))).WillOnce(Return(true));

    FilesystemCheckerGuard guard(&mock);

    auto& checker = miopen::GetFilesystemChecker();

    EXPECT_TRUE(checker.IsNetworkedFilesystem("/path1"));
    EXPECT_FALSE(checker.IsNetworkedFilesystem("/path2"));
    EXPECT_TRUE(checker.IsNetworkedFilesystem("/path3"));
}
