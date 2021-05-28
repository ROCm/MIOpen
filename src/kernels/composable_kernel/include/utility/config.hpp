#ifndef CK_CONFIG_AMD_HPP
#define CK_CONFIG_AMD_HPP

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#endif
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

// only gfx908 support native floating point atomic add
#ifndef CK_USE_AMD_BUFFER_ATOMIC_FADD
#define CK_USE_AMD_BUFFER_ATOMIC_FADD 0
#endif

#if CK_USE_AMD_BUFFER_ATOMIC_FADD
#ifndef CK_AMD_BUFFER_ATOMIC_FADD_RETURNS_FLOAT
#define CK_AMD_BUFFER_ATOMIC_FADD_RETURNS_FLOAT 0
#endif
#endif

// gfx1030 does not support V_MAD/V_MAC,but can use v_fmac_f32
#ifndef CK_USE_AMD_V_FMAC_F32
#define CK_USE_AMD_V_FMAC_F32 0
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

// block synchronization only s_wait lgkmcnt(0), not vmcnt(0)
#ifndef CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#define CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM 1
#endif

// experimental implementation
#define CK_EXPERIMENTAL_BLOCKWISE_GEMM_USE_PIPELINE 1
#define CK_EXPERIMENTAL_TENSOR_COORDINATE_USE_CALCULATE_OFFSET_DIFF 0
#define CK_EXPERIMENTAL_THREADWISE_COPY_V4R2_USE_OPTIMIZED_ADDRESS_CACLULATION 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_BLOCKWISE_GENERIC_SLICE_COPY_V1 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V1R2 0
#define CK_EXPERIMENTAL_USE_MORE_COMPILE_STATIC_THREADWISE_GENERIC_TENSOR_SLICE_COPY_V2R1 0

#ifndef CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R1_OUTPUT_SKIP_OUT_OF_BOUND_CHECK
#define CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R1_OUTPUT_SKIP_OUT_OF_BOUND_CHECK 0
#endif

#ifndef CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R1_INPUT_SKIP_OUT_OF_BOUND_CHECK
#define CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R1_INPUT_SKIP_OUT_OF_BOUND_CHECK 0
#endif

// workaround: put all workaround here
// workaround for unnecessary VGPA <--> AGRP data movement when using mfma LLVM intrinsic
#ifndef CK_WORKAROUND_SWDEV_229564
#define CK_WORKAROUND_SWDEV_229564 1
#endif
// workaround for buffer load/store fp16/bfp16 intrinsic bug
#ifndef CK_WORKAROUND_SWDEV_231101
#define CK_WORKAROUND_SWDEV_231101 1
#endif
// workaround for accvgpr over-allocation
#ifndef CK_WORKAROUND_SWDEV_241664
#define CK_WORKAROUND_SWDEV_241664 1
#endif

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

// int32x4_t used by buffer addressing LLVM intrinsic
typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));

} // namespace ck
#endif
