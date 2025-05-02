#ifndef CK_AMD_BUFFER_ADDRESSING_HPP
#define CK_AMD_BUFFER_ADDRESSING_HPP
#include "amd_buffer_intrinsics.hpp"

namespace ck {

// buffer resourse flags
/*#if defined(CK_AMD_GPU_GFX803) || defined(CK_AMD_GPU_GFX900) || defined(CK_AMD_GPU_GFX906) || \
    defined(CK_AMD_GPU_GFX942) || defined(CK_AMD_GPU_GFX908) || defined(CK_AMD_GPU_GFX90A) || \
    defined(CK_AMD_GPU_GFX950)*/
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000
/*#elif defined(CK_AMD_GPU_GFX1030) || defined(CK_AMD_GPU_GFX1031) || defined(CK_AMD_GPU_GFX1100) ||
\
    defined(CK_AMD_GPU_GFX1101) || defined(CK_AMD_GPU_GFX1102) || defined(CK_AMD_GPU_GFX1200) ||   \
    defined(CK_AMD_GPU_GFX1201)
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#endif*/

template <typename T>
__device__ buffer_resourse_t
make_raw_buffer_resourse(const T* p_src_block, short stride, int num, int flags)
{
#if CK_USE_AMD_BUFFER_PTR_TYPE
    // void *p, short stride, int num, int flags
    //  ret __amdgpu_buffer_rsrc_t
    buffer_resourse_t src_block_config = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(static_cast<const void*>(p_src_block)), stride, num, flags);
#else
    RawBufferAddressConfig<T> src_block_config_wrap;
    // 0-48 address
    // 48-64 stride and swizzle;
    src_block_config_wrap.address[0] = const_cast<T*>(p_src_block);
    src_block_config_wrap.range[1] &= (1 << 16) - 1;
    src_block_config_wrap.range[1] |= (stride << 16);
    // 64-96 records num
    src_block_config_wrap.range[2] = num;
    // 96-128
    src_block_config_wrap.range[3]     = flags;
    buffer_resourse_t src_block_config = src_block_config_wrap.data;

#endif

    return src_block_config;
}

// buffer_store requires:
//   1) p_src must be in vgpr space, d_dst must be global memory
//   2) p_dst to be a block-invariant pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ void amd_buffer_store_impl(const typename vector_type<T, N>::MemoryType src_thread_data,
                                      buffer_resourse_t dst_wave_buffer_resource,
                                      index_t dst_thread_addr_offset,
                                      index_t dst_wave_addr_offset)
{
    if constexpr(is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp32(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp32x2(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
    }
    else if constexpr(is_same<T, half_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp16(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp16x2(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp16x4(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
    }
    else if constexpr(is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_i32(src_thread_data,
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_i32x2(src_thread_data,
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_i32x4(src_thread_data,
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               0);
        }
    }
    else if constexpr(is_same<T, ushort>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_bf16(src_thread_data,
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_bf16x2(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_bf16x4(src_thread_data,
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                0);
        }
    }
    else
    {
        static_assert(false, "wrong! not implemented");
    }
}

template <typename T, index_t N>
__device__ void amd_buffer_store(const T* p_src,
                                 T* p_dst_block,
                                 const index_t dst_thread_data_offset,
                                 const index_t dst_const_data_offset)
{
    auto dst_buffer_resource = [&]() {
        // NFMT float = 7000 - ignored
        // DFMT 32 = 20000 - ignored but should not be 0
        // int32_t flag          0x00027000;
        int32_t flag = CK_BUFFER_RESOURCE_3RD_DWORD;
        // stride = stride[0:13] + bit( Cache swizzle)[14] + bit(Swizzle enable)[15]
        short stride = 0;
        int32_t num  = 0xFFFFFFFF; // max val
        return make_raw_buffer_resourse(p_dst_block, stride, num, flag);
    }();

    index_t dst_thread_addr_offset = (dst_thread_data_offset + dst_const_data_offset) * sizeof(T);

    auto typed_p_src = reinterpret_cast<const typename vector_type<T, N>::MemoryType*>(p_src);

    amd_buffer_store_impl<T, N>(*typed_p_src, dst_buffer_resource, dst_thread_addr_offset, 0);
}

// buffer_load requires:
//   1) src_wave_buffer_resource must be in global memory space, d_dst must be vgpr
//   2) src_wave_buffer_resource to be a block-invariant pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ typename vector_type<T, N>::MemoryType
amd_buffer_load_impl(buffer_resourse_t src_wave_buffer_resource,
                     index_t src_thread_addr_offset,
                     index_t src_wave_addr_offset)
{
    if constexpr(is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_fp32(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_fp32x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_fp32x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
    }
    else if constexpr(is_same<T, half_t>::value)
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_fp16(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_fp16x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_fp16x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
    }
    else if constexpr(is_same<T, ushort>::value)
    {
        if constexpr(N == 1)
        {
            return llvm_amdgcn_raw_buffer_load_bf16(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 2)
        {
            return llvm_amdgcn_raw_buffer_load_bf16x2(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
        else if constexpr(N == 4)
        {
            return llvm_amdgcn_raw_buffer_load_bf16x4(
                src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset, 0);
        }
    }
    else
    {
        static_assert(false, "wrong! not implemented");
    }
}

template <typename T, index_t VectorSize>
__device__ typename vector_type<T, VectorSize>::MemoryType
amd_buffer_load(const T* p_src_block, index_t src_thread_data_offset, index_t src_const_data_offset)
{
    // NFMT float = 7000 - ignored
    // DFMT 32 = 20000 - ignored but should not be 0
    // int32_t flag          0x00027000;
    int32_t flag = CK_BUFFER_RESOURCE_3RD_DWORD;
    // stride = stride[0:13] + bit( Cache swizzle)[14] + bit(Swizzle enable)[15]
    short stride          = 0;
    int32_t num           = 0xFFFFFFFF; // max val
    auto src_block_config = make_raw_buffer_resourse(p_src_block, stride, num, flag);

    index_t thread_addr_offset = (src_thread_data_offset + src_const_data_offset) * sizeof(T);

    return amd_buffer_load_impl<T, VectorSize>(src_block_config, thread_addr_offset, 0);
}

#if CK_USE_AMD_BUFFER_ATOMIC_FADD

template <typename T, index_t N>
__device__ void amd_buffer_atomic_add(const T* p_src,
                                      T* p_dst_block,
                                      index_t dst_thread_data_offset,
                                      index_t dst_const_data_offset)
{
    auto dst_buffer_resource = [&]() {
        // NFMT float = 7000 - ignored
        // DFMT 32 = 20000 - ignored but should not be 0
        // int32_t flag          0x00027000;
        int32_t flag = CK_BUFFER_RESOURCE_3RD_DWORD;
        // stride = stride[0:13] + bit( Cache swizzle)[14] + bit(Swizzle enable)[15]
        short stride = 0;
        int32_t num  = 0xFFFFFFFF; // max val
        return make_raw_buffer_resourse(p_dst_block, stride, num, flag);
    }();

    index_t dst_thread_addr_offset = (dst_thread_data_offset + dst_const_data_offset) * sizeof(T);

    constexpr index_t no_slc_glc = 0;

    if constexpr(is_same<T, float>::value)
    {
        for(index_t i = 0; i < N; ++i)
        {
            __llvm_amdgcn_buffer_atomic_add_f32(p_src[i],
                                                dst_buffer_resource,
                                                dst_thread_addr_offset + i * sizeof(T),
                                                0,
                                                no_slc_glc);
        }
    }
    else
    {
        static_assert(false, "wrong! not implemented");
    }
}

#endif // CK_USE_AMD_BUFFER_ATOMIC_FADD

} // namespace ck

#endif
