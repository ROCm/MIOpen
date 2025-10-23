// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// This test event listener ensures that HIP errors are cleaned up after every test, and will flag
// tests that don't clean up their own errors
class HIPErrorHandler : public testing::EmptyTestEventListener
{
    void OnTestEnd(const testing::TestInfo& test_info) override
    {
        auto hipError    = hipGetLastError();
        auto hipExtError = hipExtGetLastError();

        EXPECT_EQ(hipError, hipSuccess)
            << " hipGetLastError returned error code " << hipError << " after test "
            << test_info.test_suite_name() << "." << test_info.name()
            << ". Error string: " << hipGetErrorString(hipError);
        EXPECT_EQ(hipExtError, hipSuccess)
            << " hipExtGetLastError returned error code " << hipExtError << " after test "
            << test_info.test_suite_name() << "." << test_info.name()
            << ". Error string: " << hipGetErrorString(hipExtError);
    }
};

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new HIPErrorHandler);

    return RUN_ALL_TESTS();
}
