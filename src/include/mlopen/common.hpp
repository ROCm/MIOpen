#ifndef GUARD_MLOPEN_COMMON_HPP_
#define GUARD_MLOPEN_COMMON_HPP_

#include <mlopen.h>
#include <mlopen/manage_ptr.hpp>

#if MLOPEN_BACKEND_OPENCL

typedef cl_mem Data_t;
typedef const Data_t ConstData_t;
typedef MLOPEN_MANAGE_PTR(cl_mem, clReleaseMemObject) ManageDataPtr;

inline Data_t DataCast(void *p) {
	return (Data_t)p;
}

inline ConstData_t DataCast(const void *p) {
	return (ConstData_t)p;
}

#elif MLOPEN_BACKEND_HIP

typedef void * Data_t;
typedef const void * ConstData_t;
// TODO: Set the deleter
typedef MLOPEN_MANAGE_PTR(void, hipFree) ManageDataPtr;

inline Data_t DataCast(void *p) {
	return p;
}

inline ConstData_t DataCast(const void *p) {
	return p;
}
#endif // OpenCL vs hip
#endif // GUARD_MLOPEN_COMMON_HPP_
