#ifndef GUARD_MLOPEN_OCL_KERNEL_HPP_
#define GUARD_MLOPEN_OCL_KERNEL_HPP_

#include <sstream>
#include <vector>
#include <mlopen.h>
#include <cassert>
#include <functional>
#include <array>
#include <memory>

#include <mlopen/errors.hpp>
#include <mlopen/each_args.hpp>
#include <mlopen/clhelper.hpp>

namespace mlopen {

struct LocalMemArg 
{
	LocalMemArg(size_t _size) : size(_size) {}
	size_t GetSize() const { return size; }
	
	private:
	size_t size;
};

struct OCLSetKernelArg
{
	template<class I, class T>
	void operator()(cl_kernel kernel, I i, const T& x) const
	{
		cl_int status = clSetKernelArg(kernel, i, sizeof(T), (void *)&x);
		if (status != CL_SUCCESS) MLOPEN_THROW("Error setting argument to kernel: " + std::to_string(status));
	}

	template<class I, class T>
	void operator()(cl_kernel kernel, I i, const LocalMemArg& lmem) const
	{
		cl_int status = clSetKernelArg(kernel, i, lmem.GetSize(), NULL);
		if (status != CL_SUCCESS) MLOPEN_THROW("Error setting argument to kernel: " + std::to_string(status));
	}
};

struct OCLKernelInvoke
{
	cl_command_queue queue;
	// TODO: Use a pointer to OCLKernel
	cl_kernel kernel;
	size_t work_dim;
	std::array<size_t, 3> global_work_offset;
	std::array<size_t, 3> global_work_dim;
	std::array<size_t, 3> local_work_dim;
	std::function<void(cl_event&)> callback;

	template<class... Ts>
	void operator()(const Ts&... xs) const
	{
		each_args_i(std::bind(OCLSetKernelArg{}, kernel, std::placeholders::_1, std::placeholders::_2), xs...);
		run();
	}

	void run() const;
};

class OCLKernel {

using SharedKernelPtr = std::shared_ptr<typename std::remove_pointer<cl_kernel>::type>;

public:
	OCLKernel() {}
	OCLKernel(ClKernelPtr k) : kernel(std::move(k)) {}
	OCLKernel(ClKernelPtr k, 
			std::vector<size_t> local_dims,
			std::vector<size_t> global_dims) 
	: kernel(std::move(k)), ldims(local_dims), gdims(global_dims)
	{
		assert(ldims.size() == gdims.size());
		assert(ldims.size() > 0 && ldims.size() <= 3);
	}

	OCLKernelInvoke Invoke(cl_command_queue q, std::function<void(cl_event&)> callback=nullptr);

	cl_kernel GetKernel() { return kernel.get(); } 

	mlopenStatus_t GetKernelName(std::string &kernelName);

	inline const std::vector<size_t>& GetLocalDims() const { return ldims; }
	inline const std::vector<size_t>& GetGlobalDims() const { return gdims; }

private:
	SharedKernelPtr kernel;
	std::vector<size_t> ldims;
	std::vector<size_t> gdims;
};

}

#endif // GUARD_MLOPEN_OCL_KERNEL_HPP_
