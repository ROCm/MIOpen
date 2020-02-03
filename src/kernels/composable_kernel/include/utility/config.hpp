#ifndef CK_CONFIG_AMD_HPP
#define CK_CONFIG_AMD_HPP

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "bfloat16_dev.hpp"

// index type: unsigned or signed
#define CK_UNSIGNED_INDEX_TYPE 0

// device backend
#define CK_DEVICE_BACKEND_AMD 1

// AMD inline asm
#ifndef CK_USE_AMD_INLINE_ASM
#define CK_USE_AMD_INLINE_ASM 1
#endif

#ifndef CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM
#define CK_THREADWISE_GEMM_USE_AMD_INLINE_ASM 1
#endif

// AMD buffer addressing
#ifndef CK_USE_AMD_BUFFER_ADDRESSING
#define CK_USE_AMD_BUFFER_ADDRESSING 1
#endif

#ifndef CK_USE_AMD_BUFFER_ADDRESSING_INTRINSIC
#define CK_USE_AMD_BUFFER_ADDRESSING_INTRINSIC 1
#endif

// only support gfx908
#ifndef CK_USE_AMD_BUFFER_ATOMIC_ADD
#define CK_USE_AMD_BUFFER_ATOMIC_ADD 0
#endif

// AMD XDLOPS
#ifndef CK_USE_AMD_XDLOPS
#define CK_USE_AMD_XDLOPS 0
#endif

#ifndef CK_USE_AMD_XDLOPS_INLINE_ASM
#define CK_USE_AMD_XDLOPS_INLINE_ASM 0
#endif

#ifndef CK_USE_AMD_XDLOPS_EMULATE
#define CK_USE_AMD_XDLOPS_EMULATE 0 // For internal debug purposes
#endif

// experimental implementation
#define CK_EXPERIMENTAL_BLOCKWISE_GEMM_USE_PIPELINE 1
#define CK_EXPERIMENTAL_TENSOR_COORDINATE_USE_CALCULATE_OFFSET_DIFF 0
#define CK_EXPERIMENTAL_THREADWISE_COPY_V4R2_USE_OPTIMIZED_ADDRESS_CACLULATION 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R2 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V2R1 0

namespace ck {

enum AddressSpace
{
    Generic,
    Global,
    Lds,
    Vgpr
};

enum InMemoryDataOperation
{
    Set,
    AtomicAdd
};

#if CK_UNSIGNED_INDEX_TYPE
using index_t = uint32_t;
#else
using index_t = int32_t;
#endif

// int32x4_t use by buffer_load and buffer_store llvm intrinsic
typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));

} // namespace ck
#endif
