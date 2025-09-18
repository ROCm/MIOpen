// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef CK_AMD_BUFFER_INTRINSIC_HPP
#define CK_AMD_BUFFER_INTRINSIC_HPP

#include "static_kernel_float_type.hpp"

namespace ck {

#ifndef CK_USE_AMD_BUFFER_PTR_TYPE
#define CK_USE_AMD_BUFFER_PTR_TYPE 0
#endif // def CK_USE_AMD_BUFFER_PTR_TYPE

#if CK_USE_AMD_BUFFER_PTR_TYPE
using buffer_resourse_t = __amdgpu_buffer_rsrc_t;

#define amd_buffer_intrinsic_name(suffix_name) "llvm.amdgcn.raw.ptr.buffer." #suffix_name

#else
using buffer_resourse_t = int32x4_t;
// For 128bit SGPRs in buffer_load and buffer_store instructions
// https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
template <typename T>
union RawBufferAddressConfig
{
    int32x4_t data;
    T* address[2];
    int32_t range[4];
};

#define amd_buffer_intrinsic_name(suffix_name) "llvm.amdgcn.raw.buffer." #suffix_name

#endif // if CK_USE_AMD_BUFFER_PTR_TYPE

// load
__device__ int8_t
llvm_amdgcn_raw_buffer_load_i8(buffer_resourse_t srsrc,
                               index_t voffset,
                               index_t soffset,
                               index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.i8));

__device__ int8x2_t
llvm_amdgcn_raw_buffer_load_i8x2(buffer_resourse_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v2i8));

__device__ int8x4_t
llvm_amdgcn_raw_buffer_load_i8x4(buffer_resourse_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v4i8));

__device__ int16_t
llvm_amdgcn_raw_buffer_load_i16(buffer_resourse_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.i32));
__device__ int32_t
llvm_amdgcn_raw_buffer_load_i32(buffer_resourse_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.i32));

__device__ int32x2_t
llvm_amdgcn_raw_buffer_load_i32x2(buffer_resourse_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v2i32));

__device__ int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(buffer_resourse_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v4i32));
// half
__device__ half_t
llvm_amdgcn_raw_buffer_load_fp16(buffer_resourse_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.f16));

__device__ half2_t
llvm_amdgcn_raw_buffer_load_fp16x2(buffer_resourse_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v2f16));

__device__ half4_t
llvm_amdgcn_raw_buffer_load_fp16x4(buffer_resourse_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v4f16));

// float
__device__ float
llvm_amdgcn_raw_buffer_load_fp32(buffer_resourse_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.f32));

__device__ float2_t
llvm_amdgcn_raw_buffer_load_fp32x2(buffer_resourse_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v2f32));

__device__ float4_t
llvm_amdgcn_raw_buffer_load_fp32x4(buffer_resourse_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v4f32));

// bf16
__device__ ushort
llvm_amdgcn_raw_buffer_load_bf16(buffer_resourse_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.i16));

__device__ ushort2_t
llvm_amdgcn_raw_buffer_load_bf16x2(buffer_resourse_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v2i16));

__device__ ushort4_t
llvm_amdgcn_raw_buffer_load_bf16x4(buffer_resourse_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm(amd_buffer_intrinsic_name(load.v4i16));

// store
__device__ void
llvm_amdgcn_raw_buffer_store_i8(int8_t vdata,
                                buffer_resourse_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.i8));

__device__ void
llvm_amdgcn_raw_buffer_store_i8x2(int8x2_t vdata,
                                  buffer_resourse_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v2i8));

__device__ void
llvm_amdgcn_raw_buffer_store_i8x4(int8x4_t vdata,
                                  buffer_resourse_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v4i8));

__device__ void
llvm_amdgcn_raw_buffer_store_i16(int16_t vdata,
                                 buffer_resourse_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.i16));

__device__ void
llvm_amdgcn_raw_buffer_store_i32(int32_t vdata,
                                 buffer_resourse_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.i32));

__device__ void
llvm_amdgcn_raw_buffer_store_i32x2(int32x2_t vdata,
                                   buffer_resourse_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v2i32));

__device__ void
llvm_amdgcn_raw_buffer_store_i32x4(int32x4_t vdata,
                                   buffer_resourse_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v4i32));

// half
__device__ void
llvm_amdgcn_raw_buffer_store_fp16(half_t vdata,
                                  buffer_resourse_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.f16));

__device__ void
llvm_amdgcn_raw_buffer_store_fp16x2(half2_t vdata,
                                    buffer_resourse_t srsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v2f16));

__device__ void
llvm_amdgcn_raw_buffer_store_fp16x4(half4_t vdata,
                                    buffer_resourse_t srsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v4f16));
// float
__device__ void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  buffer_resourse_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.f32));

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x2(float2_t vdata,
                                    buffer_resourse_t srsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v2f32));

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x4(float4_t vdata,
                                    buffer_resourse_t srsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v4f32));

// float
__device__ void
llvm_amdgcn_raw_buffer_store_bf16(ushort vdata,
                                  buffer_resourse_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.i16));

__device__ void
llvm_amdgcn_raw_buffer_store_bf16x2(ushort2_t vdata,
                                    buffer_resourse_t srsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v2i16));

__device__ void
llvm_amdgcn_raw_buffer_store_bf16x4(ushort4_t vdata,
                                    buffer_resourse_t srsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm(amd_buffer_intrinsic_name(store.v4i16));

#if CK_USE_AMD_BUFFER_ATOMIC_FADD

#if CK_AMD_BUFFER_ATOMIC_FADD_RETURNS_FLOAT
__device__ float
#else
__device__ void
#endif
__llvm_amdgcn_buffer_atomic_add_f32(
    float vdata,
    buffer_resourse_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm(amd_buffer_intrinsic_name(atomic.fadd.f32));

#endif // CK_USE_AMD_BUFFER_ATOMIC_FADD

} // namespace ck
#endif // CK_AMD_BUFFER_INTRINSIC_HPP