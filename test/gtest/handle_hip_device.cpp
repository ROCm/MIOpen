// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <miopen/handle.hpp>
#include <miopen/errors.hpp>

#if MIOPEN_BACKEND_HIP
#include <hip/hip_runtime.h>

// We need access to the internal set_device and get_device_id functions
namespace miopen {
extern void set_device(int id);
extern int get_device_id();
} // namespace miopen

class CPU_HandleHipDevice_NONE : public ::testing::Test
{
protected:
    void SetUp() override
    {
        hipError_t status = hipGetDeviceCount(&device_count_);
        if(status != hipSuccess)
        {
            device_count_ = 0;
        }
    }

    void TearDown() override
    {
        // Clear any HIP error triggered by asking for an invalid device ordinal
        hipGetLastError();
        hipExtGetLastError();
    }

    int device_count_ = 0;
};

TEST_F(CPU_HandleHipDevice_NONE, ValidDeviceSetSucceeds)
{
    if(device_count_ == 0)
    {
        GTEST_SKIP() << "No HIP devices available for testing";
    }

    ASSERT_NO_THROW(miopen::set_device(0));

    int current_device = -1;
    ASSERT_EQ(hipGetDevice(&current_device), hipSuccess);
    EXPECT_EQ(current_device, 0);
}

TEST_F(CPU_HandleHipDevice_NONE, NegativeDeviceIdThrowsWithCorrectMessage)
{
    const int invalid_id = -10;

    try
    {
        miopen::set_device(invalid_id);
        FAIL() << "Expected miopen::Exception for negative device ID";
    }
    catch(const miopen::Exception& ex)
    {
        std::string error_msg = ex.what();
        EXPECT_NE(error_msg.find("Error setting device"), std::string::npos);
        EXPECT_NE(error_msg.find("-10"), std::string::npos);
    }
}

TEST_F(CPU_HandleHipDevice_NONE, OutOfRangeDeviceIdThrowsWithCorrectMessage)
{
    const int out_of_range_id = device_count_ + 10;

    try
    {
        miopen::set_device(out_of_range_id);
        FAIL() << "Expected miopen::Exception for out-of-range device ID";
    }
    catch(const miopen::Exception& ex)
    {
        std::string error_msg = ex.what();
        EXPECT_NE(error_msg.find("Error setting device"), std::string::npos);
        EXPECT_NE(error_msg.find(std::to_string(out_of_range_id)), std::string::npos);
    }
}

TEST_F(CPU_HandleHipDevice_NONE, ErrorMessageContainsDeviceIdAndStatus)
{
    const int test_device_id = 9999;

    try
    {
        miopen::set_device(test_device_id);
        FAIL() << "Expected exception for invalid device ID";
    }
    catch(const miopen::Exception& ex)
    {
        std::string msg = ex.what();
        // Verify device ID is in the message
        EXPECT_NE(msg.find("9999"), std::string::npos);
        // Verify it mentions setting device
        EXPECT_NE(msg.find("setting device"), std::string::npos);
    }
}

TEST_F(CPU_HandleHipDevice_NONE, MultipleValidDevicesSwitchCorrectly)
{
    if(device_count_ < 2)
    {
        GTEST_SKIP() << "Need at least 2 devices for multi-device TEST_F";
    }

    // Switch to device 0
    ASSERT_NO_THROW(miopen::set_device(0));
    int current = -1;
    ASSERT_EQ(hipGetDevice(&current), hipSuccess);
    EXPECT_EQ(current, 0);

    // Switch to device 1
    ASSERT_NO_THROW(miopen::set_device(1));
    ASSERT_EQ(hipGetDevice(&current), hipSuccess);
    EXPECT_EQ(current, 1);
}

TEST_F(CPU_HandleHipDevice_NONE, BoundaryConditionLastValidDevice)
{
    if(device_count_ == 0)
    {
        GTEST_SKIP() << "No devices available";
    }

    int last_valid = device_count_ - 1;
    ASSERT_NO_THROW(miopen::set_device(last_valid));

    int current = -1;
    ASSERT_EQ(hipGetDevice(&current), hipSuccess);
    EXPECT_EQ(current, last_valid);
}

TEST_F(CPU_HandleHipDevice_NONE, BoundaryConditionFirstInvalidDevice)
{
    int first_invalid = device_count_;

    try
    {
        miopen::set_device(first_invalid);
        FAIL() << "Expected exception for first invalid device ID";
    }
    catch(const miopen::Exception& ex)
    {
        std::string msg = ex.what();
        EXPECT_NE(msg.find(std::to_string(first_invalid)), std::string::npos);
    }
}

TEST_F(CPU_HandleHipDevice_NONE, GetDeviceIdReturnsValidValue)
{
    if(device_count_ == 0)
    {
        GTEST_SKIP() << "No devices available";
    }

    int device_id = miopen::get_device_id();
    EXPECT_GE(device_id, 0);
    EXPECT_LT(device_id, device_count_);
}

#else

// Dummy test for non-HIP builds
TEST(CPU_HandleHipDevice_NONE, SkippedForNonHipBackend)
{
    GTEST_SKIP() << "HIP backend not available";
}

#endif
