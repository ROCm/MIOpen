#include <mlopen/oclkernel.hpp>

namespace mlopen {

void OCLKernelInvoke::run() const
{
	cl_event ev;
	cl_int status = clEnqueueNDRangeKernel(queue, kernel,
		work_dim,
		global_work_offset.data(),
		global_work_dim.data(),
		local_work_dim.data(), 0, nullptr, callback ? &ev : nullptr);

	clFlush(queue);

	if (status != CL_SUCCESS) {
		MLOPEN_THROW("Running kernel failed: " + std::to_string(status));
	}
	else if (callback) {
		clFinish(queue);
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

mlopenStatus_t OCLKernel::GetKernelName(std::string &progName) {
	
	auto *name = new char[200];
	cl_int status = clGetKernelInfo(kernel.get(), 
			CL_KERNEL_FUNCTION_NAME, 
			200, 
			name, 
			nullptr);

	if(status != CL_SUCCESS) {
		return mlopenStatusBadParm;
	}

	progName = std::string(name);
	delete[] name;

	return mlopenStatusSuccess;
	
}

} // namespace mlopen

