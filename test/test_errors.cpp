
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
    EXPECT(strcmp(miopenGetErrorString(error), "miopenStatusSuccess") == 0);
    error = miopenStatusNotInitialized;
    EXPECT(strcmp(miopenGetErrorString(error), "miopenStatusNotInitialized") == 0);
    error = miopenStatusInvalidValue;
    EXPECT(strcmp(miopenGetErrorString(error), "miopenStatusInvalidValue") == 0);
    error = miopenStatusBadParm;
    EXPECT(strcmp(miopenGetErrorString(error), "miopenStatusBadParm") == 0);
    error = miopenStatusAllocFailed;
    EXPECT(strcmp(miopenGetErrorString(error), "miopenStatusAllocFailed") == 0);
    error = miopenStatusInternalError;
    EXPECT(strcmp(miopenGetErrorString(error), "miopenStatusInternalError") == 0);
    error = miopenStatusNotImplemented;
    EXPECT(strcmp(miopenGetErrorString(error), "miopenStatusNotImplemented") == 0);
    error = miopenStatusUnknownError;
    EXPECT(strcmp(miopenGetErrorString(error), "miopenStatusUnknownError") == 0);
}

int main()
{
    test_ocl_error();
    test_throw_cl_status();
    test_invalid_ocl_error();
    test_try();
    test_error_string();
}
