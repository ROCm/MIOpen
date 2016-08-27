#ifndef MLOPEN_GUARD_OCL_HELPER_HPP_
#define MLOPEN_GUARD_OCL_HELPER_HPP_

#include <mlopen.h>
#include <string>
#include <iostream>
#include <mlopen/manage_ptr.hpp>

namespace mlopen {

using ClProgramPtr = MLOPEN_MANAGE_PTR(cl_program, clReleaseProgram);
using ClKernelPtr = MLOPEN_MANAGE_PTR(cl_kernel, clReleaseKernel);

ClProgramPtr LoadProgram(cl_context ctx, cl_device_id device, const std::string &program_name, std::string params);
ClKernelPtr CreateKernel(cl_program program, const std::string& kernel_name);
inline ClKernelPtr CreateKernel(const ClProgramPtr& program, const std::string& kernel_name)
{
	return CreateKernel(program.get(), kernel_name);
}

cl_device_id GetDevice(cl_command_queue q);
cl_context GetContext(cl_command_queue q);
}

class CLHelper {

	public:

	static mlopenStatus_t LoadProgramFromSource(cl_program &program,
			cl_command_queue &queue,
			const std::string &program_name);

	static mlopenStatus_t BuildProgram(cl_program &program,
			cl_command_queue &queue,
			std::string params);

	static mlopenStatus_t CreateKernel(cl_program &program,
			cl_kernel &kernel,
			const std::string &kernel_name);

	static mlopenStatus_t GetDeviceFromQueue(const cl_command_queue &queue,
			cl_device_id &device);

	static mlopenStatus_t GetContextFromQueue(const cl_command_queue &queue,
			cl_context &context);

	static mlopenStatus_t CreateQueueWithProfiling(const cl_command_queue &queue,
			cl_command_queue *profile_q);

	static void CheckCLStatus(cl_int status, const std::string &desc);
};
#endif // MLOPEN_GUARD_OCL_HELPER_HPP_
