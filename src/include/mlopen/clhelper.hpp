#ifndef MLOPEN_GUARD_OCL_HELPER_HPP_
#define MLOPEN_GUARD_OCL_HELPER_HPP_

#include <mlopen.h>
#include <string>
#include <iostream>
#include <mlopen/manage_ptr.hpp>

namespace mlopen {

using ClProgramPtr = MLOPEN_MANAGE_PTR(cl_program, clReleaseProgram);
using ClKernelPtr = MLOPEN_MANAGE_PTR(cl_kernel, clReleaseKernel);
using ClAqPtr = MLOPEN_MANAGE_PTR(mlopenAcceleratorQueue_t, clReleaseCommandQueue);

ClProgramPtr LoadProgram(cl_context ctx, cl_device_id device, const std::string &program_name, std::string params, bool is_kernel_str);
ClKernelPtr CreateKernel(cl_program program, const std::string& kernel_name);
inline ClKernelPtr CreateKernel(const ClProgramPtr& program, const std::string& kernel_name)
{
	return CreateKernel(program.get(), kernel_name);
}
ClAqPtr CreateQueueWithProfiling(cl_context ctx, cl_device_id dev);

cl_device_id GetDevice(cl_command_queue q);
cl_context GetContext(cl_command_queue q);
}  // namespace mlopen

#endif // MLOPEN_GUARD_OCL_HELPER_HPP_
