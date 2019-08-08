#ifndef CK_CONFIG_AMD_HPP
#define CK_CONFIG_AMD_HPP

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"

#define CK_DEVICE_BACKEND_AMD 1
#define CK_USE_AMD_INLINE_ASM 1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1 1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1 0

namespace ck {

// For some reason, HIP compiler need this definition to generate optimal load and store
// instruction
typedef float float2_t __attribute__((ext_vector_type(2)));
typedef float float4_t __attribute__((ext_vector_type(4)));
typedef half2 half2_t;
typedef struct
{
    half2_t scalar[2];
} half4_t;
typedef struct
{
    ushort scalar[2];
} ushort2_t;
typedef struct
{
    ushort2_t scalar[2];
} ushort4_t;
using index_t = uint32_t;

__device__ void fused_multiply_accumulate(float& d, const float& s0, const float& s1)
{
    d += s0 * s1;
}

} // namespace ck

#endif
