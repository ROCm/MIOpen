#ifndef GUARD_MLOPEN_OCL_KERNEL_HPP_
#define GUARD_MLOPEN_OCL_KERNEL_HPP_

#include <sstream>
#include <vector>
#include <mlopen.h>

namespace mlopen {

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
	OCLKernel(cl_kernel k) : kernel(k) {}
	OCLKernel(cl_kernel k, 
			std::vector<size_t> local_dims,
			std::vector<size_t> gloabl_dims) : kernel(k) {
		for(int i = 0; i < ldims.size(); i++) {
			ldims.push_back(local_dims[i]);
			gdims.push_back(gloabl_dims[i]);
		}
	}

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
