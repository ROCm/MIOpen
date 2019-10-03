#ifndef CK_CONFIG_AMD_HPP
#define CK_CONFIG_AMD_HPP

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "bfloat16_dev.hpp"

#define CK_DEVICE_BACKEND_AMD 1
#define CK_USE_AMD_INLINE_ASM 1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1 1
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R1 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R2 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V2 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V2R1 0

#ifndef CK_USE_INLINE_ASM_XDLOPS
#define CK_USE_INLINE_ASM_XDLOPS 1
#endif

namespace ck {

// float
// For some reason, HIP compiler need this definition to generate optimal load and store
// instruction
typedef float float32_t __attribute__((ext_vector_type(32)));
typedef float float2_t __attribute__((ext_vector_type(2)));
typedef float float4_t __attribute__((ext_vector_type(4)));

typedef _Float16 half4_t __attribute__((ext_vector_type(4)));

typedef ushort ushort2_t __attribute__((ext_vector_type(2)));
typedef ushort ushort4_t __attribute__((ext_vector_type(4)));

// half
typedef half2 half2_t;

// index_t: used for index calculation
using index_t = uint32_t;

// data type conversion
template <class T>
struct type_convert
{
    template <class X>
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
