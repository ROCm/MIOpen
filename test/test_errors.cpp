
#include <miopen/errors.hpp>
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

void test_try()
{
    EXPECT(miopen::try_([] {}) == miopenStatusSuccess);
    EXPECT(miopen::try_([] { MIOPEN_THROW(miopenStatusInternalError); }) ==
           miopenStatusInternalError);
    EXPECT(miopen::try_([] { MIOPEN_THROW(""); }) == miopenStatusUnknownError);
    EXPECT(miopen::try_([] { throw std::runtime_error(""); }) == miopenStatusUnknownError);
    EXPECT(miopen::try_([] { throw ""; }) == miopenStatusUnknownError); // NOLINT
}

int main()
{
    test_ocl_error();
    test_invalid_ocl_error();
    test_try();
}
