#include "CLHelper.hpp"

mlopenStatus_t CLHelper::LoadProgramFromSource(cl_program &program,
		cl_command_queue &queue,
		const std::string &program_name) {

	if(queue == nullptr) {
		return mlopenStatusBadParm;
	}

	cl_int status;
	cl_context context;

	GetContextFromQueue(queue, context);

	// Stringify the kernel file
	char *source;
	size_t sourceSize;
	FILE *fp = fopen(program_name.c_str(), "rb");
	if(fp == NULL) {
		return mlopenStatusBadParm;
	}

	fseek(fp, 0, SEEK_END);
	sourceSize = ftell(fp);
	fseek(fp , 0, SEEK_SET);
	source = new char[sourceSize];
	fread(source, 1, sourceSize, fp);
	fclose(fp);

	program  = clCreateProgramWithSource(context, 
			1,
			(const char**)&source, 
			&sourceSize, 
			&status);

	CheckCLStatus(status, "Error Creating OpenCL Program (cl_program) in LoadProgramFromSource()");

	delete[] source;

	return mlopenStatusSuccess;
}

mlopenStatus_t CLHelper::BuildProgram(cl_program &program,
		cl_command_queue &queue,
		const std::string &params) {

	cl_int status;
	cl_device_id device;

	GetDeviceFromQueue(queue, device);

	/* create a cl program executable for all the devices specified */
    status = clBuildProgram(program, 
			1, &device, params.c_str(), 
			NULL, 
			NULL);

	CheckCLStatus(status, "Error Building OpenCL Program in BuildProgram()");

    if(status != CL_SUCCESS)
    {
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
	CheckCLStatus(status, error);

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

void CLHelper::CheckCLStatus(cl_int status, const std::string &errString) {
	if (status != CL_SUCCESS)
	{
		std::cout<<status<<", "<<errString<<"\n";
		exit(-1);
	}
}
