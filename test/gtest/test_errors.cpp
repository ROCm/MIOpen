/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/errors.hpp>
#include <miopen/sysinfo_utils.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

TEST(CPU_test_errors_NONE, test_ocl_error)
{
#if MIOPEN_BACKEND_OPENCL
    EXPECT_EQ(miopen::OpenCLErrorMessage(CL_SUCCESS, ":"), ": Success");
    EXPECT_EQ(miopen::OpenCLErrorMessage(3200, ":"), ":Unknown OpenCL error 3200");
    EXPECT_ANY_THROW(MIOPEN_THROW_CL_STATUS(CL_DEVICE_NOT_FOUND);));
    EXPECT_EQ(miopen::try_([] { MIOPEN_THROW_CL_STATUS(CL_DEVICE_NOT_FOUND, "OpenCL Error"); }),
              miopenStatusUnknownError);
#else
    GTEST_SKIP() << "Skipped for HIP backend";
#endif
}

TEST(CPU_test_errors_NONE, test_try)
{
    EXPECT_EQ(miopen::try_([] {}), miopenStatusSuccess);
    EXPECT_EQ(miopen::try_([] { MIOPEN_THROW(miopenStatusInternalError); }),
              miopenStatusInternalError);
    EXPECT_EQ(miopen::try_([] { MIOPEN_THROW(""); }), miopenStatusUnknownError);
    EXPECT_EQ(miopen::try_([] { throw std::runtime_error(""); }), miopenStatusUnknownError);
    EXPECT_EQ(miopen::try_([] { throw ""; }), miopenStatusUnknownError);
}

#define X_STATUS_PAIR(status) \
    std::pair { status, #status }

TEST(CPU_test_errors_NONE, test_error_string)
{
    for(auto&& [status, message] : {X_STATUS_PAIR(miopenStatusSuccess),
                                    X_STATUS_PAIR(miopenStatusNotInitialized),
                                    X_STATUS_PAIR(miopenStatusInvalidValue),
                                    X_STATUS_PAIR(miopenStatusBadParm),
                                    X_STATUS_PAIR(miopenStatusAllocFailed),
                                    X_STATUS_PAIR(miopenStatusInternalError),
                                    X_STATUS_PAIR(miopenStatusNotImplemented),
                                    X_STATUS_PAIR(miopenStatusUnknownError),
                                    X_STATUS_PAIR(miopenStatusUnsupportedOp),
                                    X_STATUS_PAIR(miopenStatusGpuOperationsSkipped),
                                    X_STATUS_PAIR(miopenStatusVersionMismatch)})
    {
        EXPECT_STREQ(miopenGetErrorString(status), message);
    }
}

TEST(CPU_test_errors_NONE, test_miopen_throw)
{
    const std::string err_msg1{"test error message"};
    const std::string err_msg2{"another error message"};
    const std::string expected_hostname = miopen::sysinfo::GetSystemHostname();

    EXPECT_THROW([&err_msg1]() { MIOPEN_THROW(err_msg1); }(), miopen::Exception);
    EXPECT_THROW([&err_msg1]() { MIOPEN_THROW(miopenStatusInternalError, err_msg1); }(),
                 miopen::Exception);

    EXPECT_THROW(
        {
            try
            {
                MIOPEN_THROW(miopenStatusUnknownError, err_msg1);
            }
            catch(const miopen::Exception& e)
            {
                EXPECT_THAT(e.message, ::testing::StartsWith(expected_hostname));
                EXPECT_THAT(e.message, ::testing::HasSubstr(__FILE__));
                EXPECT_THAT(e.message, ::testing::EndsWith(err_msg1));

                EXPECT_THAT(e.what(), ::testing::StartsWith(expected_hostname));
                EXPECT_THAT(e.what(), ::testing::HasSubstr(__FILE__));
                EXPECT_THAT(e.what(), ::testing::EndsWith(err_msg1));

                EXPECT_EQ(e.status, miopenStatusUnknownError);
                throw;
            }
        },
        miopen::Exception);

    EXPECT_THROW(
        {
            try
            {
                MIOPEN_THROW(err_msg2);
            }
            catch(const miopen::Exception& e)
            {
                EXPECT_THAT(e.message, ::testing::StartsWith(expected_hostname));
                EXPECT_THAT(e.message, ::testing::HasSubstr(__FILE__));
                EXPECT_THAT(e.message, ::testing::EndsWith(err_msg2));

                EXPECT_THAT(e.what(), ::testing::StartsWith(expected_hostname));
                EXPECT_THAT(e.what(), ::testing::HasSubstr(__FILE__));
                EXPECT_THAT(e.what(), ::testing::EndsWith(err_msg2));

                // Verify the format: hostname:file:line: message
                std::string expected_prefix = expected_hostname + ":" + __FILE__ + ":";
                EXPECT_THAT(e.message, ::testing::StartsWith(expected_prefix));

                EXPECT_EQ(e.status, miopenStatusUnknownError);
                throw;
            }
        },
        miopen::Exception);
}

TEST(CPU_test_errors_NONE, test_miopen_throw_if)
{
    const std::string err_msg1{"test error message"};
    const std::string err_msg2{"another error message"};
    const std::string expected_hostname = miopen::sysinfo::GetSystemHostname();
    const std::string fail_condition("failed condition: true");

    EXPECT_THROW([&err_msg1]() { MIOPEN_THROW_IF(true, err_msg1); }(), miopen::Exception);

    EXPECT_THROW(
        {
            try
            {
                MIOPEN_THROW_IF(true, err_msg1);
            }
            catch(const miopen::Exception& e)
            {
                EXPECT_THAT(e.message, ::testing::StartsWith(expected_hostname));
                EXPECT_THAT(e.message, ::testing::HasSubstr(__FILE__));
                EXPECT_THAT(e.message, ::testing::HasSubstr(err_msg1));
                EXPECT_THAT(e.message, ::testing::EndsWith(fail_condition));

                EXPECT_THAT(e.what(), ::testing::StartsWith(expected_hostname));
                EXPECT_THAT(e.what(), ::testing::HasSubstr(__FILE__));
                EXPECT_THAT(e.what(), ::testing::HasSubstr(err_msg1));
                EXPECT_THAT(e.what(), ::testing::EndsWith(fail_condition));

                EXPECT_EQ(e.status, miopenStatusInternalError);
                throw;
            }
        },
        miopen::Exception);

    EXPECT_THROW(
        {
            try
            {
                MIOPEN_THROW_IF(true, err_msg2);
            }
            catch(const miopen::Exception& e)
            {
                EXPECT_THAT(e.message, ::testing::StartsWith(expected_hostname));
                EXPECT_THAT(e.message, ::testing::HasSubstr(__FILE__));
                EXPECT_THAT(e.message, ::testing::HasSubstr(err_msg2));
                EXPECT_THAT(e.message, ::testing::EndsWith(fail_condition));

                EXPECT_THAT(e.what(), ::testing::StartsWith(expected_hostname));
                EXPECT_THAT(e.what(), ::testing::HasSubstr(__FILE__));
                EXPECT_THAT(e.what(), ::testing::HasSubstr(err_msg2));
                EXPECT_THAT(e.message, ::testing::EndsWith(fail_condition));

                // Verify the format: hostname:file:line: message
                std::string expected_prefix = expected_hostname + ":" + __FILE__ + ":";
                EXPECT_THAT(e.message, ::testing::StartsWith(expected_prefix));

                EXPECT_EQ(e.status, miopenStatusInternalError);
                throw;
            }
        },
        miopen::Exception);
}

TEST(CPU_test_errors_NONE, test_exception_setcontext_directly)
{
    const std::string test_file         = "test_file.cpp";
    const int test_line                 = 123;
    const std::string test_msg          = "direct context test";
    const std::string expected_hostname = miopen::sysinfo::GetSystemHostname();

    miopen::Exception ex(miopenStatusBadParm, test_msg);
    ex = ex.SetContext(test_file, test_line);

    std::string expected_message =
        expected_hostname + ":" + test_file + ":" + std::to_string(test_line) + ": " + test_msg;
    EXPECT_EQ(ex.message, expected_message);
    EXPECT_EQ(ex.status, miopenStatusBadParm);
}
