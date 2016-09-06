#ifndef GUARD_MLOPEN_DRIVER_HPP
#define GUARD_MLOPEN_DRIVER_HPP

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <mlopen.h>
#include <CL/cl.h>
#include "InputFlags.hpp"
#include <algorithm>
#include <float.h>
#include <memory>
#include <numeric>

#define UNPACK_VEC4(v) (v[0]), (v[1]), (v[2]), (v[3])

struct GPUMem {
	GPUMem() {};
	GPUMem(cl_context &ctx, size_t psz, size_t pdata_sz) : sz(psz), data_sz(pdata_sz) {	buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, data_sz*sz, NULL, NULL); }

	int ToGPU(cl_command_queue &q, void *p) { return clEnqueueWriteBuffer(q, buf, CL_TRUE, 0, data_sz*sz, p, 0, NULL, NULL); }
	int FromGPU(cl_command_queue &q, void *p) { return clEnqueueReadBuffer(q, buf, CL_TRUE, 0, data_sz*sz, p, 0, NULL, NULL); }

	cl_mem GetMem() { return buf; }

	~GPUMem() { clReleaseMemObject(buf); }

	cl_mem buf;
	size_t sz;
	size_t data_sz;
};

void Usage() {
	printf("Usage: ./driver *base_arg* *other_args*\n");
	printf("Supported Base Arguments: conv, pool, lrn, activ\n");
	exit(0);
}

std::string ParseBaseArg(int argc, char *argv[]) {
	if(argc < 2) {
		printf("Invalid Number of Input Arguments\n");
		Usage();
	}

	std::string arg = argv[1];

	if(arg != "conv" && arg != "pool" && arg != "lrn" && arg != "activ") {
		printf("Invalid Base Input Argument\n");
		Usage();
	}
	else if(arg == "-h" || arg == "--help" || arg == "-?")
		Usage();
	else
		return arg;

	return 0;
}

class Driver
{
	public: 
	Driver() {
		mlopenCreate(&handle);
		mlopenGetStream(handle, &q);
	}

	mlopenHandle_t GetHandle() { return handle; }
	cl_command_queue& GetStream() { return q; }

	virtual ~Driver() {
		mlopenDestroy(handle);
	}

	// TODO: add timing APIs
	virtual int AddCmdLineArgs() = 0;
	virtual int ParseCmdLineArgs(int argc, char *argv[]) = 0;
	virtual InputFlags & GetInputFlags() = 0;
	virtual int GetandSetData() = 0;
	virtual int AllocateBuffersAndCopy() = 0;
	virtual int RunForwardGPU() = 0;
	virtual int VerifyForward() = 0;
	virtual int RunBackwardGPU() = 0;
	virtual int VerifyBackward() = 0;

	protected:

	mlopenHandle_t handle;
	cl_command_queue q;
};

#endif // GUARD_MLOPEN_DRIVER_HPP
