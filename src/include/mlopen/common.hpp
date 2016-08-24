#ifndef GUARD_MLOPEN_COMMON_HPP_
#define GUARD_MLOPEN_COMMON_HPP_

#include <mlopen.h>
#include <mlopen/manage_ptr.hpp>

template<class T>
auto tie2(T&& x) -> decltype(std::tie(x[0], x[1]))
{
	return std::tie(x[0], x[1]);
}

#if MLOPEN_BACKEND_OPENCL

typedef cl_mem Data_t;
typedef MLOPEN_MANAGE_PTR(cl_mem, clReleaseMemObject) ManageDataPtr;

inline Data_t DataCast(void *p) {
	return (Data_t)p;
}

inline const Data_t DataCast(const void *p) {
	return (Data_t)p;
}

#elif MLOPEN_BACKEND_HIP

typedef void * Data_t;
// TODO: Set the deleter
typedef std::unique_ptr<void> ManageDataPtr;

inline Data_t DataCast(void *p) {
	return p;
}

inline const Data_t DataCast(const void *p) {
	return p;
}
#endif // OpenCL vs hip
#endif // GUARD_MLOPEN_COMMON_HPP_
