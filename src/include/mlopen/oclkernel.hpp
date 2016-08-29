#ifndef GUARD_MLOPEN_OCL_KERNEL_HPP_
#define GUARD_MLOPEN_OCL_KERNEL_HPP_

#include <sstream>
#include <array>
#include <vector>
#include <mlopen.h>
#include <cassert>
#include <functional>

#include <mlopen/errors.hpp>
#include <mlopen/each_args.hpp>

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

	public:
	OCLKernel() {}
	OCLKernel(cl_kernel k) : kernel(k) {}
	OCLKernel(cl_kernel k, 
			std::vector<size_t> local_dims,
			std::vector<size_t> global_dims) 
	: kernel(k), ldims(local_dims), gdims(global_dims) 
	{
		assert(ldims.size() == gdims.size());
		assert(ldims.size() > 0 && ldims.size() <= 3);
	}

	OCLKernelInvoke Invoke(cl_command_queue q, std::function<void(cl_event&)> callback=nullptr);

	//TODO: when to call the destructor?
//	~OCLKernel() { clReleaseKernel(_kernel); }

	template<typename T, typename... Args>
	mlopenStatus_t SetArgs(int i, const T& first, const Args&... rest);
	template<typename... Args>
	mlopenStatus_t SetArgs(int i, const LocalMemArg &lmem, const Args&... rest);
	mlopenStatus_t SetArgs(int) {
		return mlopenStatusSuccess;
	}

	mlopenStatus_t run(cl_command_queue &queue,
		const int &work_dim,
		const size_t * global_work_offset,
		const size_t * global_work_dim,
		const size_t * local_work_dim,
		cl_event	 * event);

	cl_kernel& GetKernel() { return kernel; } 

	mlopenStatus_t GetKernelName(std::string &kernelName);

	inline const std::vector<size_t>& GetLocalDims() const { return ldims; }
	inline const std::vector<size_t>& GetGlobalDims() const { return gdims; }

private:
	cl_kernel kernel;
	std::vector<size_t> ldims;
	std::vector<size_t> gdims;
};

template<typename T, typename... Args>
mlopenStatus_t OCLKernel::SetArgs(int i, 
		const T& first, 
		const Args&... rest)
{
	cl_int status;

	status = clSetKernelArg(kernel, i++, sizeof(T), (void *)& first);
	std::stringstream errStream;
	errStream<<"OpenCL error setting kernel argument "<<i;
//	clCheckStatus(status, errStream.str()) ;

	status = SetArgs(i, rest...);
	return mlopenStatusSuccess;

}

template<typename... Args>
mlopenStatus_t OCLKernel::SetArgs(int i, 
		const LocalMemArg &lmem, 
		const Args&... rest)
{
	cl_int status;
	status = clSetKernelArg(kernel, i++, lmem.GetSize(), NULL);
	std::stringstream errStream;
	errStream<<"OpenCL error setting kernel argument (local memory) "<<i;
	//clCheckStatus(status, errStream.str()) ;
	
	status = SetArgs(i, rest...);
	return mlopenStatusSuccess;

}

}

#endif // GUARD_MLOPEN_OCL_KERNEL_HPP_
