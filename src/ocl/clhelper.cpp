#include "clhelper.hpp"
#include <mlopen/kernel.hpp>
#include <mlopen/errors.hpp>

mlopenStatus_t CLHelper::LoadProgramFromSource(cl_program &program,
		cl_command_queue &queue,
		const std::string &program_name) {

	if(queue == nullptr) {
		return mlopenStatusBadParm;
	}

	cl_int status;
	cl_context context;

	GetContextFromQueue(queue, context);

	std::string source = mlopen::GetKernelSrc(program_name);

	const char* char_source = source.c_str();
	auto size = source.size();

	program  = clCreateProgramWithSource(context, 
			1,
			(const char**)&char_source, 
			&size, 
			&status);

	CheckCLStatus(status, "Error Creating OpenCL Program (cl_program) in LoadProgramFromSource()");

	return mlopenStatusSuccess;
}

mlopenStatus_t CLHelper::BuildProgram(cl_program &program,
		cl_command_queue &queue,
		std::string params) {

	// Temporary hack to properly add the flags without causing spacing problems or duplication
	// MD: I do not think the path is required here anyways, it is only required to find the kernel
	// which we are doing in LoadProgramFromSource.
	//
	// Also, removing the CL2.0 flag for now due to incorrect code generation found by Alex
	// params += " -cl-std=CL2.0";

	cl_int status;
	cl_device_id device;

	GetDeviceFromQueue(queue, device);

	/* create a cl program executable for all the devices specified */
    status = clBuildProgram(program, 
			1, &device, params.c_str(), 
			NULL, 
			NULL);

	// MD: CheckCLStatus exits but we need to get the ProgramBuildInfo.
//	CheckCLStatus(status, "Error Building OpenCL Program in BuildProgram()");

    if(status != CL_SUCCESS)
    {
		printf("Error Building OpenCL Program in BuildProgram()\n");
        char * errorbuf = (char*)calloc(sizeof(char),1024*1024);
        size_t size;
        clGetProgramBuildInfo(program,
				device,
				CL_PROGRAM_BUILD_LOG, 
				1024*1024, 
				errorbuf,
				&size);

        printf("%s ", errorbuf);
		free(errorbuf);
		return mlopenStatusBadParm;
    }

    return mlopenStatusSuccess;

}

mlopenStatus_t CLHelper::CreateKernel(cl_program &program,
		cl_kernel &kernel,
		const std::string &kernel_name) {

	if(program == nullptr) {
		mlopenStatusBadParm;
	}

	cl_int status;

	kernel = clCreateKernel(program, 
			kernel_name.c_str(), 
			&status);

	std::string error = "Error Creating Kernel [" + kernel_name + "] in CreateKernel()";
	if(status != CL_SUCCESS) {
		std::cout<<error<<" "<<status<<"\n";
		return mlopenStatusBadParm;
	}
	// MD: Cannot use CheckCLStatus because the search needs to continue even if one
	// kernel fails. Rather, the search should be graceful not to create a kernel
	// if conditions are not met
	//CheckCLStatus(status, error);

	return mlopenStatusSuccess;
}

mlopenStatus_t CLHelper::GetDeviceFromQueue(const cl_command_queue &queue,
		cl_device_id &device) {

	cl_int status;

	status = clGetCommandQueueInfo(queue,
			CL_QUEUE_DEVICE, 
			sizeof(cl_device_id),
			&device, 
			NULL);

	CheckCLStatus(status, "Error Getting Device Info from Queue in GetDecviceFromQueue()");
	return mlopenStatusSuccess;
}

mlopenStatus_t CLHelper::GetContextFromQueue(const cl_command_queue &queue,
		cl_context &context) {

	cl_int status;

	status = clGetCommandQueueInfo(queue,
			CL_QUEUE_CONTEXT, 
			sizeof(cl_context),
			&context, 
			NULL);

	CheckCLStatus(status, "Error Getting Device Info from Queue in GetDecviceFromQueue()");
	return mlopenStatusSuccess;
}

mlopenStatus_t CLHelper::CreateQueueWithProfiling(const cl_command_queue &queue,
		cl_command_queue *profile_q) {

	cl_device_id dev;
	cl_context ctx;
	cl_int status;

	GetContextFromQueue(queue, ctx);
	GetDeviceFromQueue(queue, dev);

	*profile_q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &status);

	CheckCLStatus(status, "Error Creating Queue With Profiling Enabled");

	return mlopenStatusSuccess;
}

void CLHelper::CheckCLStatus(cl_int status, const std::string &errString) {
	if (status != CL_SUCCESS)
	{
		MLOPEN_THROW(errString + std::to_string(status));
	}
}
