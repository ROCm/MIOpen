#ifndef GUARD_MLOPEN_COMMON_HPP_
#define GUARD_MLOPEN_COMMON_HPP_

#include <mlopen.h>
#include <mlopen/manage_ptr.hpp>

#if MLOPEN_BACKEND_OPENCL

using Data_t = cl_mem;
// Const doesnt apply to cl_mem
using ConstData_t = Data_t;
using ManageDataPtr = MLOPEN_MANAGE_PTR(cl_mem, clReleaseMemObject);

inline Data_t DataCast(void *p) {
	return reinterpret_cast<Data_t>(p);
}

inline ConstData_t DataCast(const void *p) {
// Casting away const is undefined behaviour, but we do it anyways
#ifdef MLOPEN_USE_CLANG_TIDY
    static cl_mem s = nullptr;
    (void)p;
    return s;
#else
	return reinterpret_cast<ConstData_t>(const_cast<void*>(p));
#endif
}

#elif MLOPEN_BACKEND_HIP || MLOPEN_BACKEND_HIPOC

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
