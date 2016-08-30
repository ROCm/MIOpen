#ifndef GUARD_MLOPEN_COMMON_HPP_
#define GUARD_MLOPEN_COMMON_HPP_

#include <mlopen.h>
#include <mlopen/manage_ptr.hpp>

#if MLOPEN_BACKEND_OPENCL

using Data_t = cl_mem;
using ConstData_t = const Data_t;
using ManageDataPtr = mlopen::manage_ptr<typename std::remove_pointer<cl_mem>::type, decltype(&clReleaseMemObject), &clReleaseMemObject>;

inline Data_t DataCast(void *p) {
	return reinterpret_cast<Data_t>(p);
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
