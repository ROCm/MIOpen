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
                                           bool slc) __asm("llvm.amdgcn.buffer.load");

__device__ float2_t __llvm_amdgcn_buffer_loadx2(int32x4_t rsrc,
                                                index_t vindex,
                                                index_t offset,
                                                bool glc,
                                                bool slc) __asm("llvm.amdgcn.buffer.load.dwordx2");

__device__ float4_t __llvm_amdgcn_buffer_loadx4(int32x4_t rsrc,
                                                index_t vindex,
                                                index_t offset,
                                                bool glc,
                                                bool slc) __asm("llvm.amdgcn.buffer.load.dwordx4");

__device__ void __llvm_amdgcn_buffer_store(float vdata,
                                           int32x4_t rsrc,
                                           index_t vindex,
                                           index_t offset,
                                           bool glc,
                                           bool slc) __asm("llvm.amdgcn.buffer.store");

__device__ void __llvm_amdgcn_buffer_storex2(float2_t vdata,
                                             int32x4_t rsrc,
                                             index_t vindex,
                                             index_t offset,
                                             bool glc,
                                             bool slc) __asm("llvm.amdgcn.buffer.store.dwordx2");

__device__ void __llvm_amdgcn_buffer_storex4(float4_t vdata,
                                             int32x4_t rsrc,
                                             index_t vindex,
                                             index_t offset,
                                             bool glc,
                                             bool slc) __asm("llvm.amdgcn.buffer.store.dwordx4");

template <typename T, index_t VectorSize>
__device__ typename vector_type<T, VectorSize>::MemoryType
__buffer_load(const T* p_src_block, index_t src_thread_data_offset, index_t src_const_data_offset);

template <typename T, index_t VectorSize>
__device__ void __buffer_store(const typename vector_type<T, VectorSize>::MemoryType& src,
                               T* p_dst_block,
                               index_t dst_thread_data_offset,
                               index_t dst_const_data_offset);

template <>
__device__ float __buffer_load<float, 1>(const float* p_src_block,
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
__device__ float2_t __buffer_load<float, 2>(const float* p_src_block,
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
__device__ float4_t __buffer_load<float, 4>(const float* p_src_block,
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
__device__ void __buffer_store<float, 1>(const float& src,
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
    __llvm_amdgcn_buffer_store(src,
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
                   "v"(src),
                   "v"(dst_thread_addr_offset),
                   "s"(dst_const_addr_offset));
#endif
}

template <>
__device__ void __buffer_store<float, 2>(const float2_t& src,
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
    __llvm_amdgcn_buffer_storex2(src,
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
                   "v"(src),
                   "v"(dst_thread_addr_offset),
                   "s"(dst_const_addr_offset));
#endif
}

template <>
__device__ void __buffer_store<float, 4>(const float4_t& src,
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
    __llvm_amdgcn_buffer_storex4(src,
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
                   "v"(src),
                   "v"(dst_thread_addr_offset),
                   "s"(dst_const_addr_offset));
#endif
}

} // namespace ck
#endif
