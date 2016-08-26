#include <mlopen/oclkernel.hpp>

namespace mlopen {

void OCLKernelInvoke::run() const
{
	cl_event ev;
	cl_int status = clEnqueueNDRangeKernel(queue, kernel,
		work_dim,
		global_work_offset.data(),
		global_work_dim.data(),
		local_work_dim.data(), 0, NULL, callback ? &ev : nullptr);

	if (status != CL_SUCCESS) {
		MLOPEN_THROW("Running kernel failed: " + std::to_string(status));
	}
	else if (callback) {
		clWaitForEvents(1, &ev);
		callback(ev);
	}
}

OCLKernelInvoke OCLKernel::Invoke(cl_command_queue q, std::function<void(cl_event&)> callback)
{
	OCLKernelInvoke result{q, kernel.get(), ldims.size(), {}, {}, {}, callback};
	std::copy(gdims.begin(), gdims.end(), result.global_work_dim.begin());
	std::copy(ldims.begin(), ldims.end(), result.local_work_dim.begin());
	return result;
}

mlopenStatus_t OCLKernel::run(cl_command_queue &queue,
	const int &work_dim,
	const size_t  * global_work_offset,
	const size_t  * global_work_dim,
	const size_t  * local_work_dim,
	cl_event	  * event) {

	cl_int status = clEnqueueNDRangeKernel(queue, kernel.get(),
		work_dim,
		global_work_offset,
		global_work_dim,
		local_work_dim, 0, NULL, event);

	//TODO: Check for error 
	if (status != CL_SUCCESS) {
		printf("kernel failed %d \n",status);
	}
	else if (event) {
		clWaitForEvents(1, event);
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t OCLKernel::GetKernelName(std::string &progName) {
	
	char *name = new char[200];
	cl_int status = clGetKernelInfo(kernel.get(), 
			CL_KERNEL_FUNCTION_NAME, 
			200, 
			name, 
			NULL);

	if(status != CL_SUCCESS) {
		return mlopenStatusBadParm;
	}

	progName = std::string(name);
	delete[] name;

	return mlopenStatusSuccess;
	
}

}

