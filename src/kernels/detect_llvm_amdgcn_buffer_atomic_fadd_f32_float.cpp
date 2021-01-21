#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#endif

typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));

__device__ float
__llvm_amdgcn_buffer_atomic_add_f32(float vdata,
                                    int32x4_t rsrc,
                                    int32_t vindex,
                                    int32_t offset,
                                    bool slc) __asm("llvm.amdgcn.buffer.atomic.fadd.f32");

extern "C" __global__ void test_llvm_amdgcn_buffer_atomic_fadd_f32_float(float* p_global)
{
    int32x4_t buffer_resource{0};

    float tmp = __llvm_amdgcn_buffer_atomic_add_f32(*p_global, buffer_resource, 0, 0, false);

    p_global[0] = tmp;
}
