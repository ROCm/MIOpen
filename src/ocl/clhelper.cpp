#include <fstream>
#include <mlopen/clhelper.hpp>
#include <mlopen/kernel.hpp>
#include <mlopen/errors.hpp>

namespace mlopen {

ClProgramPtr LoadProgram(cl_context ctx, cl_device_id device, const std::string &program_name, std::string params, bool is_binary)
{
	std::string source = mlopen::GetKernelSrc(program_name);

	const char* char_source = source.c_str();
	auto size = source.size();
	cl_int status, binaryStatus;
	ClProgramPtr result;

	if (is_binary)
	{
		result = ClProgramPtr{clCreateProgramWithBinary(ctx,
			1,
			&device,
			reinterpret_cast<const size_t*>(&size),
			reinterpret_cast<const unsigned char**>(&char_source),
			&status,
			&binaryStatus) };

		if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, "Error creating code object program (cl_program) in LoadProgramFromBinary()"); }
	}
	else
	{
		result = ClProgramPtr{clCreateProgramWithSource(ctx,
				1,
				&char_source, 
				&size, 
				&status)};
		if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, "Error Creating OpenCL Program (cl_program) in LoadProgram()"); }
	}

	params += " -cl-std=CL2.0";
	status = clBuildProgram(result.get(), 
			1, &device, params.c_str(), 
			nullptr, 
			nullptr);

	if(status != CL_SUCCESS)
    {
		std::string msg = "Error Building OpenCL Program in BuildProgram()\n";
		std::vector<char> errorbuf(1024*1024);
        size_t psize;
        clGetProgramBuildInfo(result.get(),
				device,
				CL_PROGRAM_BUILD_LOG, 
				1024*1024, 
				errorbuf.data(),
				&psize);

		msg += errorbuf.data();
		if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, msg); }
    }

	return result;

}
ClKernelPtr CreateKernel(cl_program program, const std::string& kernel_name)
{
	cl_int status;
	ClKernelPtr result{clCreateKernel(program, 
			kernel_name.c_str(), 
			&status)};

	if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status); }

	return result;
}

cl_device_id GetDevice(cl_command_queue q)
{
	cl_device_id device;
	cl_int status = clGetCommandQueueInfo(q,
			CL_QUEUE_DEVICE, 
			sizeof(cl_device_id),
			&device, 
			nullptr);
	if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, "Error Getting Device Info from Queue in GetDevice()"); }

	return device;
}

cl_context GetContext(cl_command_queue q)
{
	cl_context context;
	cl_int status = clGetCommandQueueInfo(q,
			CL_QUEUE_CONTEXT, 
			sizeof(cl_context),
			&context, 
			nullptr);
	if (status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status, "Error Getting Device Info from Queue in GetDevice()"); }
	return context;
}

ClAqPtr CreateQueueWithProfiling(cl_context ctx, cl_device_id dev) 
{
	cl_int status;
	ClAqPtr q{clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &status)};

	if(status != CL_SUCCESS) { MLOPEN_THROW_CL_STATUS(status); }

	return q;
}

} // namespace mlopen
