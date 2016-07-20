#ifndef _MLOPEN_HANDLE_HPP_
#define _MLOPEN_HANDLE_HPP_

#include "MLOpen.h"
#include <vector>
#include <cstdio>
#include <cstring>

// TODO: Should be here and not in MLOpen.h
#if MLOpen_BACKEND_OPENCL
//#include <CL/cl.h>
//typedef cl_command_queue mlopenStream_t;

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#elif MLOpen_BACKEND_HIP
//#include <hip_runtime.h>
//typedef hipStream_t mlopenStream_t;

#endif // OpenCL or HIP

struct mlopenContext {
	
	mlopenContext() {};
	mlopenContext(int numStreams, mlopenStream_t *streams);
	~mlopenContext() {};

	template <typename Stream>
	mlopenStatus_t CreateDefaultStream();
	mlopenStatus_t SetStream(int numStreams, mlopenStream_t *streams);
	mlopenStatus_t GetStream(mlopenStream_t *stream, int numStream = 0) const;
	
	std::vector<mlopenStream_t> _streams;
	bool use_default_stream = false;
};

#if MLOpen_BACKEND_OPENCL
template<>
mlopenStatus_t mlopenContext::CreateDefaultStream<cl_command_queue>();
#elif MLOpen_BACKEND_HIP
template<>
mlopenStatus_t mlopenContext::CreateDefaultStream<hipStream_t>();
#endif // HIP vs OpenCL

#endif // _MLOPEN_HANDLE_HPP_
