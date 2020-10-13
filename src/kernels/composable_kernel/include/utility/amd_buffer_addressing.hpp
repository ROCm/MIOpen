#ifndef CK_AMD_BUFFER_ADDRESSING_HPP
#define CK_AMD_BUFFER_ADDRESSING_HPP

#include "float_type.hpp"

namespace ck {

// For 128bit SGPRs in buffer_load and buffer_store instructions
// https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
template <typename T>
union BufferAddressConfig
{
    int32x4_t data;
    T* address[2];
    int32_t range[4];
};

__device__ float __llvm_amdgcn_buffer_load_f32(int32x4_t rsrc,
                                               index_t vindex,
                                               index_t offset,
                                               bool glc,
                                               bool slc) __asm("llvm.amdgcn.buffer.load.f32");

__device__ float2_t
__llvm_amdgcn_buffer_load_f32x2(int32x4_t rsrc,
                                index_t vindex,
                                index_t offset,
                                bool glc,
                                bool slc) __asm("llvm.amdgcn.buffer.load.v2f32");

__device__ float4_t
__llvm_amdgcn_buffer_load_f32x4(int32x4_t rsrc,
                                index_t vindex,
                                index_t offset,
                                bool glc,
                                bool slc) __asm("llvm.amdgcn.buffer.load.v4f32");

__device__ half_t
__llvm_amdgcn_raw_buffer_load_f16(int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16");

__device__ ushort
__llvm_amdgcn_raw_buffer_load_bf16(int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.bf16");

__device__ void __llvm_amdgcn_buffer_store_f32(float vdata,
                                               int32x4_t rsrc,
                                               index_t vindex,
                                               index_t offset,
                                               bool glc,
                                               bool slc) __asm("llvm.amdgcn.buffer.store.f32");

__device__ void __llvm_amdgcn_buffer_store_f32x2(float2_t vdata,
                                                 int32x4_t rsrc,
                                                 index_t vindex,
                                                 index_t offset,
                                                 bool glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v2f32");

__device__ void __llvm_amdgcn_buffer_store_f32x4(float4_t vdata,
                                                 int32x4_t rsrc,
                                                 index_t vindex,
                                                 index_t offset,
                                                 bool glc,
                                                 bool slc) __asm("llvm.amdgcn.buffer.store.v4f32");

__device__ void
__llvm_amdgcn_raw_buffer_store_f16(half_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f16");

__device__ void
__llvm_amdgcn_raw_buffer_store_bf16(ushort vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.bf16");

#if CK_USE_AMD_BUFFER_ATOMIC_FADD
#if CK_HIP_VERSION_FLAT >= 3010020405
// starting ROCm-3.10, the return type becomes float
__device__ float
#else
__device__ void
#endif
__llvm_amdgcn_buffer_atomic_add_f32(float vdata,
                                    int32x4_t rsrc,
                                    index_t vindex,
                                    index_t offset,
                                    bool slc) __asm("llvm.amdgcn.buffer.atomic.fadd.f32");
#endif

// buffer_load requires:
//   1) p_src must be in global memory space, d_dst must be vgpr
//   2) p_src to be a block-invariant pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ typename vector_type<T, VectorSize>::MemoryType amd_buffer_load(
    const T* p_src_block, index_t src_thread_data_offset, index_t src_const_data_offset);

// buffer_store requires:
//   1) p_src must be in vgpr space, d_dst must be global memory
//   2) p_dst to be a block-invariant pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ void amd_buffer_store(const T* p_src,
                                 T* p_dst_block,
                                 index_t dst_thread_data_offset,
                                 index_t dst_const_data_offset);

#if CK_USE_AMD_BUFFER_ATOMIC_FADD
template <typename T, index_t VectorSize>
__device__ void amd_buffer_atomic_add(const T* p_src,
                                      T* p_dst_block,
                                      index_t dst_thread_data_offset,
                                      index_t dst_const_data_offset);
#endif

template <>
__device__ float amd_buffer_load<float, 1>(const float* p_src_block,
                                           index_t src_thread_data_offset,
                                           index_t src_const_data_offset)
{
    BufferAddressConfig<float> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<float*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    return __llvm_amdgcn_buffer_load_f32(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);
}

template <>
__device__ float2_t amd_buffer_load<float, 2>(const float* p_src_block,
                                              index_t src_thread_data_offset,
                                              index_t src_const_data_offset)
{
    BufferAddressConfig<float> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<float*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    return __llvm_amdgcn_buffer_load_f32x2(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);
}

template <>
__device__ float4_t amd_buffer_load<float, 4>(const float* p_src_block,
                                              index_t src_thread_data_offset,
                                              index_t src_const_data_offset)
{
    BufferAddressConfig<float> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<float*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    return __llvm_amdgcn_buffer_load_f32x4(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);
}

template <>
__device__ half_t amd_buffer_load<half_t, 1>(const half_t* p_src_block,
                                             index_t src_thread_data_offset,
                                             index_t src_const_data_offset)
{
    BufferAddressConfig<half_t> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<half_t*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(half_t);

    // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
    // everything is passed to Voffset
    return __llvm_amdgcn_raw_buffer_load_f16(
        src_block_config.data, src_thread_addr_offset + src_const_addr_offset, 0, 0);
}

template <>
__device__ half2_t amd_buffer_load<half_t, 2>(const half_t* p_src_block,
                                              index_t src_thread_data_offset,
                                              index_t src_const_data_offset)
{
    BufferAddressConfig<half_t> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<half_t*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(half_t);

    float dst_out_tmp = __llvm_amdgcn_buffer_load_f32(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);

    return *reinterpret_cast<half2_t*>(&dst_out_tmp);
}

template <>
__device__ half4_t amd_buffer_load<half_t, 4>(const half_t* p_src_block,
                                              index_t src_thread_data_offset,
                                              index_t src_const_data_offset)
{
    BufferAddressConfig<half_t> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<half_t*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(half_t);

    float2_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x2(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);

    return *reinterpret_cast<half4_t*>(&dst_out_tmp);
}

template <>
__device__ half8_t amd_buffer_load<half_t, 8>(const half_t* p_src_block,
                                              index_t src_thread_data_offset,
                                              index_t src_const_data_offset)
{
    BufferAddressConfig<half_t> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<half_t*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(half_t);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(half_t);

    float4_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x4(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);

    return *reinterpret_cast<half8_t*>(&dst_out_tmp);
}

template <>
__device__ ushort amd_buffer_load<ushort, 1>(const ushort* p_src_block,
                                             index_t src_thread_data_offset,
                                             index_t src_const_data_offset)
{
    BufferAddressConfig<ushort> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<ushort*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(ushort);

    // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
    // everything is passed to Voffset
    return __llvm_amdgcn_raw_buffer_load_bf16(
        src_block_config.data, src_thread_addr_offset + src_const_addr_offset, 0, 0);
}

template <>
__device__ ushort2_t amd_buffer_load<ushort, 2>(const ushort* p_src_block,
                                                index_t src_thread_data_offset,
                                                index_t src_const_data_offset)
{
    BufferAddressConfig<ushort> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<ushort*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(ushort);

    float dst_out_tmp = __llvm_amdgcn_buffer_load_f32(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);

    return *reinterpret_cast<ushort2_t*>(&dst_out_tmp);
}

template <>
__device__ ushort4_t amd_buffer_load<ushort, 4>(const ushort* p_src_block,
                                                index_t src_thread_data_offset,
                                                index_t src_const_data_offset)
{
    BufferAddressConfig<ushort> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<ushort*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(ushort);

    float2_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x2(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);

    return *reinterpret_cast<ushort4_t*>(&dst_out_tmp);
}

template <>
__device__ ushort8_t amd_buffer_load<ushort, 8>(const ushort* p_src_block,
                                                index_t src_thread_data_offset,
                                                index_t src_const_data_offset)
{
    BufferAddressConfig<ushort> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<ushort*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(ushort);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(ushort);

    float4_t dst_out_tmp = __llvm_amdgcn_buffer_load_f32x4(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);

    return *reinterpret_cast<ushort8_t*>(&dst_out_tmp);
}

template <>
__device__ void amd_buffer_store<float, 1>(const float* p_src,
                                           float* p_dst_block,
                                           index_t dst_thread_data_offset,
                                           index_t dst_const_data_offset)
{
    BufferAddressConfig<float> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    __llvm_amdgcn_buffer_store_f32(*p_src,
                                   dst_block_config.data,
                                   0,
                                   dst_thread_addr_offset + dst_const_addr_offset,
                                   false,
                                   false);
}

template <>
__device__ void amd_buffer_store<float, 2>(const float* p_src,
                                           float* p_dst_block,
                                           index_t dst_thread_data_offset,
                                           index_t dst_const_data_offset)
{
    BufferAddressConfig<float> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    __llvm_amdgcn_buffer_store_f32x2(*reinterpret_cast<const float2_t*>(p_src),
                                     dst_block_config.data,
                                     0,
                                     dst_thread_addr_offset + dst_const_addr_offset,
                                     false,
                                     false);
}

template <>
__device__ void amd_buffer_store<float, 4>(const float* p_src,
                                           float* p_dst_block,
                                           index_t dst_thread_data_offset,
                                           index_t dst_const_data_offset)
{
    BufferAddressConfig<float> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    __llvm_amdgcn_buffer_store_f32x4(*reinterpret_cast<const float4_t*>(p_src),
                                     dst_block_config.data,
                                     0,
                                     dst_thread_addr_offset + dst_const_addr_offset,
                                     false,
                                     false);
}

template <>
__device__ void amd_buffer_store<half_t, 1>(const half_t* p_src,
                                            half_t* p_dst_block,
                                            index_t dst_thread_data_offset,
                                            index_t dst_const_data_offset)
{
    BufferAddressConfig<half_t> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(half_t);

    // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
    // everything is passed to Voffset
    __llvm_amdgcn_raw_buffer_store_f16(
        *p_src, dst_block_config.data, dst_thread_addr_offset + dst_const_addr_offset, 0, 0);
}

template <>
__device__ void amd_buffer_store<half_t, 2>(const half_t* p_src,
                                            half_t* p_dst_block,
                                            index_t dst_thread_data_offset,
                                            index_t dst_const_data_offset)
{
    BufferAddressConfig<half_t> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(half_t);

    const float* p_src_tmp = reinterpret_cast<const float*>(p_src);

    __llvm_amdgcn_buffer_store_f32(*p_src_tmp,
                                   dst_block_config.data,
                                   0,
                                   dst_thread_addr_offset + dst_const_addr_offset,
                                   false,
                                   false);
}

template <>
__device__ void amd_buffer_store<half_t, 4>(const half_t* p_src,
                                            half_t* p_dst_block,
                                            index_t dst_thread_data_offset,
                                            index_t dst_const_data_offset)
{
    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(half_t);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(half_t);

    BufferAddressConfig<half_t> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    const float2_t* p_src_tmp = reinterpret_cast<const float2_t*>(p_src);

    __llvm_amdgcn_buffer_store_f32x2(*p_src_tmp,
                                     dst_block_config.data,
                                     0,
                                     dst_thread_addr_offset + dst_const_addr_offset,
                                     false,
                                     false);
}

template <>
__device__ void amd_buffer_store<ushort, 1>(const ushort* p_src,
                                            ushort* p_dst_block,
                                            index_t dst_thread_data_offset,
                                            index_t dst_const_data_offset)
{
    BufferAddressConfig<ushort> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(ushort);

    // current code cannot isolate Soffset and Voffset, so Soffset is hard-coded to 0, and
    // everything is passed to Voffset
    __llvm_amdgcn_raw_buffer_store_bf16(
        *p_src, dst_block_config.data, dst_thread_addr_offset + dst_const_addr_offset, 0, 0);
}

template <>
__device__ void amd_buffer_store<ushort, 2>(const ushort* p_src,
                                            ushort* p_dst_block,
                                            index_t dst_thread_data_offset,
                                            index_t dst_const_data_offset)
{
    BufferAddressConfig<ushort> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(ushort);

    const float* p_src_tmp = reinterpret_cast<const float*>(p_src);

    __llvm_amdgcn_buffer_store_f32(*p_src_tmp,
                                   dst_block_config.data,
                                   0,
                                   dst_thread_addr_offset + dst_const_addr_offset,
                                   false,
                                   false);
}

template <>
__device__ void amd_buffer_store<ushort, 4>(const ushort* p_src,
                                            ushort* p_dst_block,
                                            index_t dst_thread_data_offset,
                                            index_t dst_const_data_offset)
{
    BufferAddressConfig<ushort> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(ushort);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(ushort);

    const float2_t* p_src_tmp = reinterpret_cast<const float2_t*>(p_src);

    __llvm_amdgcn_buffer_store_f32x2(*p_src_tmp,
                                     dst_block_config.data,
                                     0,
                                     dst_thread_addr_offset + dst_const_addr_offset,
                                     false,
                                     false);
}

#if CK_USE_AMD_BUFFER_ATOMIC_FADD
template <>
__device__ void amd_buffer_atomic_add<float, 1>(const float* p_src,
                                                float* p_dst_block,
                                                index_t dst_thread_data_offset,
                                                index_t dst_const_data_offset)
{
    BufferAddressConfig<float> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    __llvm_amdgcn_buffer_atomic_add_f32(
        *p_src, dst_block_config.data, 0, dst_thread_addr_offset + dst_const_addr_offset, false);
}

template <>
__device__ void amd_buffer_atomic_add<float, 2>(const float* p_src,
                                                float* p_dst_block,
                                                index_t dst_thread_data_offset,
                                                index_t dst_const_data_offset)
{
    for(index_t i = 0; i < 2; ++i)
    {
        amd_buffer_atomic_add<float, 1>(
            &p_src[i], p_dst_block, dst_thread_data_offset, dst_const_data_offset + i);
    }
}

template <>
__device__ void amd_buffer_atomic_add<float, 4>(const float* p_src,
                                                float* p_dst_block,
                                                index_t dst_thread_data_offset,
                                                index_t dst_const_data_offset)
{
    for(index_t i = 0; i < 4; ++i)
    {
        amd_buffer_atomic_add<float, 1>(
            &p_src[i], p_dst_block, dst_thread_data_offset, dst_const_data_offset + i);
    }
}
#endif // CK_USE_AMD_BUFFER_ATOMIC_FADD

} // namespace ck
#endif
