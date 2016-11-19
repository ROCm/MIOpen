#include <mlopen/oclkernel.hpp>

namespace mlopen {

void OCLKernelInvoke::run() const
{
	cl_event ev;
	cl_int status = clEnqueueNDRangeKernel(queue, kernel.get(),
		work_dim,
		global_work_offset.data(),
		global_work_dim.data(),
		local_work_dim.data(), 0, nullptr, callback ? &ev : nullptr);

	clFlush(queue);

	if (status != CL_SUCCESS) {
		MLOPEN_THROW_CL_STATUS(status, "Running kernel failed: ");
	}
	else if (callback) {
		clFinish(queue);
		clWaitForEvents(1, &ev);
		callback(ev);
	}
}

OCLKernelInvoke OCLKernel::Invoke(cl_command_queue q, std::function<void(cl_event&)> callback)
{
	OCLKernelInvoke result{q, kernel, ldims.size(), {}, {}, {}, callback};
	std::copy(gdims.begin(), gdims.end(), result.global_work_dim.begin());
	std::copy(ldims.begin(), ldims.end(), result.local_work_dim.begin());
	return result;
}

std::string OCLKernel::GetName() const
{
	std::array<char, 200> buffer{};

	cl_int status = clGetKernelInfo(kernel.get(), 
			CL_KERNEL_FUNCTION_NAME, 
			200, 
			buffer.data(), 
			nullptr);

	if(status != CL_SUCCESS) 
	{
		MLOPEN_THROW_CL_STATUS(status, "Error getting kernel name");
	}
	return buffer.data();
}

} // namespace mlopen

