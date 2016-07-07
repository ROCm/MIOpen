#include "OCLKernel.hpp"

mlopenStatus_t OCLKernel::run(cl_command_queue &queue,
	const int &work_dim,
	const size_t &global_work_offset,
	const size_t &global_work_dim,
	const size_t &local_work_dim) {

	cl_int status = clEnqueueNDRangeKernel(queue, _kernel,
		work_dim,
		&global_work_offset,
		&global_work_dim,
		&local_work_dim, 0, NULL, NULL);

	//TODO: Check for error 
	if (status != CL_SUCCESS) {
		printf("kernelfailed\n");
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t OCLKernel::run(cl_command_queue &queue,
	const int &work_dim,
	const size_t  * global_work_offset,
	const size_t  * global_work_dim,
	const size_t  * local_work_dim) {

	cl_int status = clEnqueueNDRangeKernel(queue, _kernel,
		work_dim,
		global_work_offset,
		global_work_dim,
		local_work_dim, 0, NULL, NULL);

	//TODO: Check for error 
	if (status != CL_SUCCESS) {
		printf("kernelfailed\n");
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t OCLKernel::GetKernelName(std::string &progName) {
	
	char *name = new char[200];
	cl_int status = clGetKernelInfo(_kernel, 
			CL_KERNEL_FUNCTION_NAME, 
			200, 
			name, 
			NULL);

	if(status != CL_SUCCESS) {
		return mlopenStatusBadParm;
	}

	progName = std::string(name);
	free(name);

	return mlopenStatusSuccess;
	
}

