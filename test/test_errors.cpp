
#include <miopen/errors.hpp>
#include <cstring>
#include "test.hpp"

void test_ocl_error()
{
#if MIOPEN_BACKEND_OPENCL
    EXPECT(miopen::OpenCLErrorMessage(CL_SUCCESS, ":") == ": Success");
#endif
}

void test_invalid_ocl_error()
{
#if MIOPEN_BACKEND_OPENCL
    EXPECT(miopen::OpenCLErrorMessage(3200, ":") == ":Unknown OpenCL error 3200");
#endif
}

void test_throw_cl_status()
{
#if MIOPEN_BACKEND_OPENCL
    EXPECT(throws([] { MIOPEN_THROW_CL_STATUS(CL_DEVICE_NOT_FOUND); }));
    EXPECT(miopen::try_([] { MIOPEN_THROW_CL_STATUS(CL_DEVICE_NOT_FOUND, "OpenCL Error"); }) ==
           miopenStatusUnknownError);
#endif
}

void test_try()
{
    EXPECT(miopen::try_([] {}) == miopenStatusSuccess);
    EXPECT(miopen::try_([] { MIOPEN_THROW(miopenStatusInternalError); }) ==
           miopenStatusInternalError);
    EXPECT(miopen::try_([] { MIOPEN_THROW(""); }) == miopenStatusUnknownError);
    EXPECT(miopen::try_([] { throw std::runtime_error(""); }) == miopenStatusUnknownError);
    EXPECT(miopen::try_([] { throw ""; }) == miopenStatusUnknownError); // NOLINT
}

void test_error_string()
{
    miopenStatus_t error = miopenStatusSuccess;
    EXPECT_EQUAL(std::string(miopenGetErrorString(error)), "miopenStatusSuccess");
    error = miopenStatusNotInitialized;
    EXPECT_EQUAL(std::string(miopenGetErrorString(error)), "miopenStatusNotInitialized");
    error = miopenStatusInvalidValue;
    EXPECT_EQUAL(std::string(miopenGetErrorString(error)), "miopenStatusInvalidValue");
    error = miopenStatusBadParm;
    EXPECT_EQUAL(std::string(miopenGetErrorString(error)), "miopenStatusBadParm");
    error = miopenStatusAllocFailed;
    EXPECT_EQUAL(std::string(miopenGetErrorString(error)), "miopenStatusAllocFailed");
    error = miopenStatusInternalError;
    EXPECT_EQUAL(std::string(miopenGetErrorString(error)), "miopenStatusInternalError");
    error = miopenStatusNotImplemented;
    EXPECT_EQUAL(std::string(miopenGetErrorString(error)), "miopenStatusNotImplemented");
    error = miopenStatusUnknownError;
    EXPECT_EQUAL(std::string(miopenGetErrorString(error)), "miopenStatusUnknownError");
}

int main()
{
    test_ocl_error();
    test_throw_cl_status();
    test_invalid_ocl_error();
    test_try();
    test_error_string();
}
