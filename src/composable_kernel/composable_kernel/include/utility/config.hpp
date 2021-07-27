#ifndef CK_CONFIG_AMD_HPP
#define CK_CONFIG_AMD_HPP

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#endif
#include "bfloat16_dev.hpp"

// address space for kernel parameter
#define CONSTANT __attribute__((address_space(4)))

// GPU target
// should enable one and only one GPU target
#if !(defined(CK_AMD_GPU_GFX803) || defined(CK_AMD_GPU_GFX900) || defined(CK_AMD_GPU_GFX906) || \
      defined(CK_AMD_GPU_GFX908) || defined(CK_AMD_GPU_GFX90A) || defined(CK_AMD_GPU_GFX1030))
#error Need to define a single GPU target
#endif

// HIP version
#ifndef CK_HIP_VERSION_FLAT
#define CK_HIP_VERSION_FLAT 0
#endif

// launch bounds
#define CK_USE_LAUNCH_BOUNDS 1

#ifdef CK_USE_LAUNCH_BOUNDS
#define CK_MAX_THREAD_PER_BLOCK 256
#define CK_MIN_BLOCK_PER_CU 2
#endif

// buffer resourse
#if defined(CK_AMD_GPU_GFX803) || defined(CK_AMD_GPU_GFX900) || defined(CK_AMD_GPU_GFX906) || \
    defined(CK_AMD_GPU_GFX908) || defined(CK_AMD_GPU_GFX90A)
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(CK_AMD_GPU_GFX1030)
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#endif

// multi index
#define CK_USE_DYNAMICALLY_INDEXED_MULTI_INDEX 0

// AMD inline asm
#ifndef CK_USE_AMD_INLINE_ASM
#define CK_USE_AMD_INLINE_ASM 1
#endif

// AMD DLOPS
#ifndef CK_USE_AMD_DLOP
#define CK_USE_AMD_DLOP 1
#endif

#ifndef CK_USE_AMD_DLOP_INLINE_ASM
#define CK_USE_AMD_DLOP_INLINE_ASM 1
#endif

// AMD buffer addressing
#ifndef CK_USE_AMD_BUFFER_ADDRESSING
#define CK_USE_AMD_BUFFER_ADDRESSING 1
#endif

// only gfx908 support native floating point atomic add
#ifndef CK_USE_AMD_BUFFER_ATOMIC_FADD
#define CK_USE_AMD_BUFFER_ATOMIC_FADD 0
#endif

// AMD XDLOPS
#ifndef CK_USE_AMD_XDLOPS
#define CK_USE_AMD_XDLOPS 0
#endif

// block synchronization only s_wait lgkmcnt(0), not vmcnt(0)
#ifndef CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#define CK_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM 1
#endif

// experimental implementation
#ifndef CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
#define CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 0
#endif

#ifndef CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
#define CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_OOB_CHECK_OFFSET_TRICK
#define CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_OOB_CHECK_OFFSET_TRICK 1
#endif

// pass tensor descriptor by value or void*
#define CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE 0
#define CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER 1

// merge transformation use magic number division
#define CK_EXPERIMENTAL_MERGE_USE_MAGIC_DIVISION 0

// hack: have underlying assumption that need to be satsified, otherwise it's a bug
// hack for forcing register to keep idx_diff_low_const in SGPR. idx_diff_low_const must be
// thread-invariant, otherwise it's a bug
// TODO: separate index calculation into "compile-time", "global", "block", "wave", "thread"
#ifndef CK_HACK_DYNAMIC_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
#define CK_HACK_DYNAMIC_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE 0
#endif

// workaround for compiler crash when compiling recursive lambda
#ifndef CK_WORKAROUND_SWDEV_275126
#define CK_WORKAROUND_SWDEV_275126 1
#endif

// workaround for compiler crash when using buffer load/store for i8
#ifndef CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE
#define CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE 1
#endif

// workaround for compiler crash when using buffer load/store for i8
#ifndef CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
#define CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE 1
#endif

namespace ck {

enum AddressSpaceEnum_t
{
    Generic,
    Global,
    Lds,
    Sgpr,
    Vgpr
};

enum InMemoryDataOperationEnum_t
{
    Set,
    AtomicAdd
};

// index type
using index_t = int32_t;

} // namespace ck
#endif
