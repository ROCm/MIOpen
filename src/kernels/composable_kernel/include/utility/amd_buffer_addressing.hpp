#ifndef CK_AMD_BUFFER_ADDRESSING_HPP
#define CK_AMD_BUFFER_ADDRESSING_HPP

#include "float_type.hpp"

namespace ck {

// For 128bit SGPRs in buffer_load and buffer_store instructions
// https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
template <typename T>
union BufferLoadStoreDwordConfig
{
    int32x4_t data;
    T* address[2];
    int32_t range[4];
};

__device__ float __llvm_amdgcn_buffer_load(int32x4_t rsrc,
                                           index_t vindex,
                                           index_t offset,
                                           bool glc,
                                           bool slc) __asm("llvm.amdgcn.buffer.load.f32");

__device__ float2_t __llvm_amdgcn_buffer_loadx2(int32x4_t rsrc,
                                                index_t vindex,
                                                index_t offset,
                                                bool glc,
                                                bool slc) __asm("llvm.amdgcn.buffer.load.v2f32");

__device__ float4_t __llvm_amdgcn_buffer_loadx4(int32x4_t rsrc,
                                                index_t vindex,
                                                index_t offset,
                                                bool glc,
                                                bool slc) __asm("llvm.amdgcn.buffer.load.v4f32");

__device__ void __llvm_amdgcn_buffer_store(float vdata,
                                           int32x4_t rsrc,
                                           index_t vindex,
                                           index_t offset,
                                           bool glc,
                                           bool slc) __asm("llvm.amdgcn.buffer.store.f32");

__device__ void __llvm_amdgcn_buffer_storex2(float2_t vdata,
                                             int32x4_t rsrc,
                                             index_t vindex,
                                             index_t offset,
                                             bool glc,
                                             bool slc) __asm("llvm.amdgcn.buffer.store.v2f32");

__device__ void __llvm_amdgcn_buffer_storex4(float4_t vdata,
                                             int32x4_t rsrc,
                                             index_t vindex,
                                             index_t offset,
                                             bool glc,
                                             bool slc) __asm("llvm.amdgcn.buffer.store.v4f32");

__device__ void
__llvm_amdgcn_buffer_atomic_add(float vdata,
                                int32x4_t rsrc,
                                index_t vindex,
                                index_t offset,
                                bool slc) __asm("llvm.amdgcn.buffer.atomic.fadd.f32");

// buffer_load requires:
//   1) p_src must be in global memory space, d_dst must be vgpr
//   2) p_src to be a block-invariant pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ typename vector_type<T, VectorSize>::MemoryType amd_intrinsic_buffer_load(
    const T* p_src_block, index_t src_thread_data_offset, index_t src_const_data_offset);

// buffer_store requires:
//   1) p_src must be in vgpr space, d_dst must be global memory
//   2) p_dst to be a block-invariant pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t VectorSize>
__device__ void amd_intrinsic_buffer_store(const T* p_src,
                                           T* p_dst_block,
                                           index_t dst_thread_data_offset,
                                           index_t dst_const_data_offset);

template <typename T, index_t VectorSize>
__device__ void amd_intrinsic_buffer_atomic_add(const T* p_src,
                                                T* p_dst_block,
                                                index_t dst_thread_data_offset,
                                                index_t dst_const_data_offset);

template <>
__device__ float amd_intrinsic_buffer_load<float, 1>(const float* p_src_block,
                                                     index_t src_thread_data_offset,
                                                     index_t src_const_data_offset)
{
    float dst;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    BufferLoadStoreDwordConfig<float> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<float*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

#if CK_USE_AMD_BUFFER_ADDRESSING_INTRINSIC
    dst = __llvm_amdgcn_buffer_load(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);
#else
    asm volatile(
        "\n \
    buffer_load_dword %0, %1, %2, %3 offen offset:0 \n \
    s_waitcnt 0 \n \
    "
        : "=v"(dst)
        : "v"(src_thread_addr_offset), "s"(src_block_config.data), "s"(src_const_addr_offset));
#endif

    return dst;
}

template <>
__device__ float2_t amd_intrinsic_buffer_load<float, 2>(const float* p_src_block,
                                                        index_t src_thread_data_offset,
                                                        index_t src_const_data_offset)
{
    float2_t dst;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    BufferLoadStoreDwordConfig<float> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<float*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

#if CK_USE_AMD_BUFFER_ADDRESSING_INTRINSIC
    dst = __llvm_amdgcn_buffer_loadx2(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);
#else
    asm volatile(
        "\n \
    buffer_load_dwordx2 %0, %1, %2, %3 offen offset:0 \n \
    s_waitcnt 0 \n \
    "
        : "=v"(dst)
        : "v"(src_thread_addr_offset), "s"(src_block_config.data), "s"(src_const_addr_offset));
#endif

    return dst;
}

template <>
__device__ float4_t amd_intrinsic_buffer_load<float, 4>(const float* p_src_block,
                                                        index_t src_thread_data_offset,
                                                        index_t src_const_data_offset)
{
    float4_t dst;

    index_t src_thread_addr_offset = src_thread_data_offset * sizeof(float);
    index_t src_const_addr_offset  = src_const_data_offset * sizeof(float);

    BufferLoadStoreDwordConfig<float> src_block_config;

    // fill in byte 0 - 1
    src_block_config.address[0] = const_cast<float*>(p_src_block);
    // fill in byte 2
    src_block_config.range[2] = -1;
    // fill in byte 3
    src_block_config.range[3] = 0x00027000;

#if CK_USE_AMD_BUFFER_ADDRESSING_INTRINSIC
    dst = __llvm_amdgcn_buffer_loadx4(
        src_block_config.data, 0, src_thread_addr_offset + src_const_addr_offset, false, false);
#else
    asm volatile(
        "\n \
    buffer_load_dwordx4 %0, %1, %2, %3 offen offset:0 \n \
    s_waitcnt 0 \n \
    "
        : "=v"(dst)
        : "v"(src_thread_addr_offset), "s"(src_block_config.data), "s"(src_const_addr_offset));
#endif

    return dst;
}

template <>
__device__ void amd_intrinsic_buffer_store<float, 1>(const float* p_src,
                                                     float* p_dst_block,
                                                     index_t dst_thread_data_offset,
                                                     index_t dst_const_data_offset)
{
    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    BufferLoadStoreDwordConfig<float> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

#if CK_USE_AMD_BUFFER_ADDRESSING_INTRINSIC
    __llvm_amdgcn_buffer_store(*p_src,
                               dst_block_config.data,
                               0,
                               dst_thread_addr_offset + dst_const_addr_offset,
                               false,
                               false);
#else
    asm volatile("\n \
    buffer_store_dword %1, %2, %0, %3 offen offset:0 \n \
    "
                 :
                 : "s"(dst_block_config.data),
                   "v"(*p_src),
                   "v"(dst_thread_addr_offset),
                   "s"(dst_const_addr_offset));
#endif
}

template <>
__device__ void amd_intrinsic_buffer_store<float, 2>(const float* p_src,
                                                     float* p_dst_block,
                                                     index_t dst_thread_data_offset,
                                                     index_t dst_const_data_offset)
{
    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    BufferLoadStoreDwordConfig<float> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

#if CK_USE_AMD_BUFFER_ADDRESSING_INTRINSIC
    __llvm_amdgcn_buffer_storex2(*reinterpret_cast<const float2_t*>(p_src),
                                 dst_block_config.data,
                                 0,
                                 dst_thread_addr_offset + dst_const_addr_offset,
                                 false,
                                 false);
#else
    asm volatile("\n \
    buffer_store_dwordx2 %1, %2, %0, %3 offen offset:0 \n \
    "
                 :
                 : "s"(dst_block_config.data),
                   "v"(*reinterpret_cast<const float2_t*>(p_src)),
                   "v"(dst_thread_addr_offset),
                   "s"(dst_const_addr_offset));
#endif
}

template <>
__device__ void amd_intrinsic_buffer_store<float, 4>(const float* p_src,
                                                     float* p_dst_block,
                                                     index_t dst_thread_data_offset,
                                                     index_t dst_const_data_offset)
{
    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    BufferLoadStoreDwordConfig<float> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

#if CK_USE_AMD_BUFFER_ADDRESSING_INTRINSIC
    __llvm_amdgcn_buffer_storex4(*reinterpret_cast<const float4_t*>(p_src),
                                 dst_block_config.data,
                                 0,
                                 dst_thread_addr_offset + dst_const_addr_offset,
                                 false,
                                 false);
#else
    asm volatile("\n \
    buffer_store_dwordx4 %1, %2, %0, %3 offen offset:0 \n \
    "
                 :
                 : "s"(dst_block_config.data),
                   "v"(*reinterpret_cast<const float4_t*>(p_src)),
                   "v"(dst_thread_addr_offset),
                   "s"(dst_const_addr_offset));
#endif
}

template <>
__device__ void amd_intrinsic_buffer_atomic_add<float, 1>(const float* p_src,
                                                          float* p_dst_block,
                                                          index_t dst_thread_data_offset,
                                                          index_t dst_const_data_offset)
{
    index_t dst_thread_addr_offset = dst_thread_data_offset * sizeof(float);
    index_t dst_const_addr_offset  = dst_const_data_offset * sizeof(float);

    BufferLoadStoreDwordConfig<float> dst_block_config;

    // fill in byte 0 - 1
    dst_block_config.address[0] = p_dst_block;
    // fill in byte 2
    dst_block_config.range[2] = -1;
    // fill in byte 3
    dst_block_config.range[3] = 0x00027000;

#if CK_USE_AMD_BUFFER_ADDRESSING_INTRINSIC
    __llvm_amdgcn_buffer_atomic_add(
        *p_src, dst_block_config.data, 0, dst_thread_addr_offset + dst_const_addr_offset, false);
#else
    static_assert(false, " wrong! not implemented");
#endif
}

template <>
__device__ void amd_intrinsic_buffer_atomic_add<float, 2>(const float* p_src,
                                                          float* p_dst_block,
                                                          index_t dst_thread_data_offset,
                                                          index_t dst_const_data_offset)
{
    for(index_t i = 0; i < 2; ++i)
    {
        amd_intrinsic_buffer_atomic_add<float, 1>(
            &p_src[i], p_dst_block, dst_thread_data_offset, dst_const_data_offset + i);
    }
}

template <>
__device__ void amd_intrinsic_buffer_atomic_add<float, 4>(const float* p_src,
                                                          float* p_dst_block,
                                                          index_t dst_thread_data_offset,
                                                          index_t dst_const_data_offset)
{
    for(index_t i = 0; i < 4; ++i)
    {
        amd_intrinsic_buffer_atomic_add<float, 1>(
            &p_src[i], p_dst_block, dst_thread_data_offset, dst_const_data_offset + i);
    }
}

} // namespace ck
#endif
