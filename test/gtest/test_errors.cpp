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
#include <gtest/gtest.h>

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
