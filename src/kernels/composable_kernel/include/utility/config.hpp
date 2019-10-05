#ifndef CK_CONFIG_AMD_HPP
#define CK_CONFIG_AMD_HPP

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "bfloat16_dev.hpp"

// device backend
#define CK_DEVICE_BACKEND_AMD 1

// AMD inline asm
#ifndef CK_USE_AMD_INLINE_ASM
#define CK_USE_AMD_INLINE_ASM 1
#endif

#ifndef CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM
#define CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM 1
#endif

// AMD XDLOPS
#ifndef CK_USE_AMD_XDLOPS
#define CK_USE_AMD_XDLOPS 1
#endif

#ifndef CK_USE_AMD_XDLOPS_INLINE_ASM
#define CK_USE_AMD_XDLOPS_INLINE_ASM 1
#endif

// experimental implementation
#define CK_EXPERIMENTAL_BLOCKWISE_GEMM_USE_PIPELINE 1

#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1 1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R2 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V2R1 0

// workaround
#define CK_WORKAROUND_SWDEV_202749 1

namespace ck {

// index_t: used for index calculation
using index_t = uint32_t;

// For some reason, HIP compiler need this definition to generate optimal load and store
//   instruction
// float
typedef float float2_t __attribute__((ext_vector_type(2)));
typedef float float4_t __attribute__((ext_vector_type(4)));
typedef float float32_t __attribute__((ext_vector_type(32)));

// float16
typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));

// bfloat16
typedef ushort ushort2_t __attribute__((ext_vector_type(2)));
typedef ushort ushort4_t __attribute__((ext_vector_type(4)));

// data type conversion
template <typename T>
struct type_convert
{
    template <typename X>
    __device__ T operator()(X x) const
    {
        return static_cast<T>(x);
    }
};

template <>
template <>
__device__ float type_convert<float>::operator()<ushort>(ushort x) const
{
    return bfloat16_to_float(x);
}

template <>
template <>
__device__ ushort type_convert<ushort>::operator()<float>(float x) const
{
    return float_to_bfloat16(x);
}

} // namespace ck
#endif
