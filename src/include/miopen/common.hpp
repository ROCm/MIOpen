#ifndef GUARD_MIOPEN_COMMON_HPP_
#define GUARD_MIOPEN_COMMON_HPP_

#include <miopen.h>
#include <miopen/manage_ptr.hpp>

#if MIOPEN_BACKEND_OPENCL

using Data_t = cl_mem;
// Const doesnt apply to cl_mem
using ConstData_t = Data_t;
using ManageDataPtr = MIOPEN_MANAGE_PTR(cl_mem, clReleaseMemObject);

inline Data_t DataCast(void *p) {
	return reinterpret_cast<Data_t>(p);
}

inline ConstData_t DataCast(const void *p) {
// Casting away const is undefined behaviour, but we do it anyways
#ifdef MIOPEN_USE_CLANG_TIDY
    static cl_mem s = nullptr;
    (void)p;
    return s;
#else
	return reinterpret_cast<ConstData_t>(const_cast<void*>(p));
#endif
}

#elif MIOPEN_BACKEND_HIP || MIOPEN_BACKEND_HIPOC

using Data_t = void *;
using ConstData_t = const void *;
// TODO: Set the deleter
using ManageDataPtr = MIOPEN_MANAGE_PTR(void, hipFree);

inline Data_t DataCast(void *p) {
	return p;
}

inline ConstData_t DataCast(const void *p) {
	return p;
}
#endif // OpenCL vs hip
#endif // GUARD_MIOPEN_COMMON_HPP_
