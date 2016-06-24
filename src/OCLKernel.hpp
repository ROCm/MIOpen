#ifndef _OCL_KERNEL_HPP_
#define _OCL_KERNEL_HPP_

#include <sstream>
#include "MLOpen.h"

struct LocalMemArg 
{
	LocalMemArg(size_t _size) : size(_size) {}
	size_t GetSize() const { return size; }
	
	private:
	size_t size;
};

class OCLKernel {

	public:
	OCLKernel() {}
	OCLKernel(cl_kernel kernel) : _kernel(kernel) {}

	//TODO: when to call the destructor?
//	~OCLKernel() { clReleaseKernel(_kernel); }

	template<typename T, typename... Args>
	mlopenStatus_t SetArgs(int i, const T& first, const Args&... rest);
	template<typename... Args>
	mlopenStatus_t SetArgs(int i, const LocalMemArg &lmem, const Args&... rest);
	mlopenStatus_t SetArgs(int i) {}

	mlopenStatus_t run(cl_command_queue &queue,
			const int &work_dim,
			const size_t &global_work_offset,
			const size_t &global_work_dim,
			const size_t &local_work_dim);

	cl_kernel& GetKernel() { return _kernel; } 

	mlopenStatus_t GetKernelName(std::string &kernelName);

	private:
	cl_kernel _kernel;
};

template<typename T, typename... Args>
mlopenStatus_t OCLKernel::SetArgs(int i, 
		const T& first, 
		const Args&... rest)
{
	cl_int status;

	status = clSetKernelArg(_kernel, i++, sizeof(T), (void *)& first);
	std::stringstream errStream;
	errStream<<"OpenCL error setting kernel argument "<<i;
//	clCheckStatus(status, errStream.str()) ;

	SetArgs(i, rest...);
}

template<typename... Args>
mlopenStatus_t OCLKernel::SetArgs(int i, 
		const LocalMemArg &lmem, 
		const Args&... rest)
{
	cl_int status;
	status = clSetKernelArg(_kernel, i++, lmem.GetSize(), NULL);
	std::stringstream errStream;
	errStream<<"OpenCL error setting kernel argument (local memory) "<<i;
	//clCheckStatus(status, errStream.str()) ;
	
	SetArgs(i, rest...);

}

#endif // _OCL_KERNEL_HPP_
