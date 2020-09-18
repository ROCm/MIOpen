#ifndef CK_IN_MEMORY_OPERATION_AMD_HPP
#define CK_IN_MEMORY_OPERATION_AMD_HPP

#include "float_type.hpp"

#if CK_USE_AMD_BUFFER_ADDRESSING
#include "amd_buffer_addressing.hpp"
#endif

namespace ck {

template <typename T>
__device__ void atomic_add_impl(T* p_dst, T src)
{
    atomicAdd(p_dst, src);
}

// atomicAdd for float does not support vector type
template <>
__device__ void atomic_add_impl<float2_t>(float2_t* p_dst, float2_t src)
{
    float* p_dst_float       = reinterpret_cast<float*>(p_dst);
    const float* p_src_float = reinterpret_cast<const float*>(&src);

    for(index_t i = 0; i < 2; ++i)
    {
        atomicAdd(&(p_dst_float[i]), p_src_float[i]);
    }
}

template <>
__device__ void atomic_add_impl<float4_t>(float4_t* p_dst, float4_t src)
{
    float* p_dst_float       = reinterpret_cast<float*>(p_dst);
    const float* p_src_float = reinterpret_cast<const float*>(&src);

    for(index_t i = 0; i < 4; ++i)
    {
        atomicAdd(&(p_dst_float[i]), p_src_float[i]);
    }
}

template <typename T, index_t DataPerAccess>
struct SetData
{
    using vector_t = typename vector_type<T, DataPerAccess>::MemoryType;

    // This version is only for compatibility, don't use this version if possible
    template <AddressSpace SrcAddressSpace, AddressSpace DstAddressSpace>
    __device__ void Run(const T* p_src, index_t src_offset, T* p_dst, index_t dst_offset) const
    {
        *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) =
            *reinterpret_cast<const vector_t*>(&p_src[src_offset]);
    }

#if CK_USE_AMD_BUFFER_ADDRESSING
    // buffer_load requires:
    //   1) p_src must be in global memory space, d_dst must be vgpr
    //   2) p_src to be a block-invariant pointer.
    // It is user's responsibility to make sure that is true.
    template <>
    __device__ void Run<AddressSpace::Global, AddressSpace::Vgpr>(const T* p_src,
                                                                  index_t src_offset,
                                                                  T* p_dst,
                                                                  index_t dst_offset) const
    {
        *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) =
            amd_buffer_load<T, DataPerAccess>(p_src, src_offset, 0);
    }

    // buffer_store requires:
    //   1) p_src must be in vgpr space, d_dst must be global memory
    //   2) p_dst to be a block-invariant pointer.
    // It is user's responsibility to make sure that is true.
    template <>
    __device__ void Run<AddressSpace::Vgpr, AddressSpace::Global>(const T* p_src,
                                                                  index_t src_offset,
                                                                  T* p_dst,
                                                                  index_t dst_offset) const
    {
        amd_buffer_store<T, DataPerAccess>(&(p_src[src_offset]), p_dst, dst_offset, 0);
    }
#endif
};

template <typename T, index_t DataPerAccess>
struct AtomicAddData
{
    using vector_t = typename vector_type<T, DataPerAccess>::MemoryType;

    // This version is only for compatibility, don't use this version if possible
    template <AddressSpace SrcAddressSpace, AddressSpace DstAddressSpace>
    __device__ void Run(const T* p_src, index_t src_offset, T* p_dst, index_t dst_offset) const
    {
        atomic_add_impl(reinterpret_cast<vector_t*>(&p_dst[dst_offset]),
                        *reinterpret_cast<const vector_t*>(&p_src[src_offset]));
    }

#if CK_USE_AMD_BUFFER_ADDRESSING && CK_USE_AMD_BUFFER_ATOMIC_ADD
    // buffer_atomic_add requires:
    //   1) p_src must be in vgpr space, d_dst must be global memory
    //   2) p_dst to be a block-invariant pointer.
    // It is user's responsibility to make sure that is true.
    template <>
    __device__ void Run<AddressSpace::Vgpr, AddressSpace::Global>(const T* p_src,
                                                                  index_t src_offset,
                                                                  T* p_dst,
                                                                  index_t dst_offset) const
    {
        amd_buffer_atomic_add<T, DataPerAccess>(&(p_src[src_offset]), p_dst, dst_offset, 0);
    }
#endif
};

template <typename T,
          index_t DataPerAccess,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t SrcDataStride = 1,
          index_t DstDataStride = 1>
__device__ void transfer_data(const T* p_src, index_t src_offset, T* p_dst, index_t dst_offset)
{
    static_assert(DstInMemOp == InMemoryDataOperation::Set ||
                      DstInMemOp == InMemoryDataOperation::AtomicAdd,
                  "wrong! InMemoryDataOperation not supported!");

    // keep it simple, don't use static_if here, otherwise compiler will do weird things
    if(SrcDataStride == 1 && DstDataStride == 1)
    {
        // TODO: use static_if::ElseIf
        static_if<DstInMemOp == InMemoryDataOperation::Set>{}([&](auto) {
            SetData<T, DataPerAccess>{}.template Run<SrcAddressSpace, DstAddressSpace>(
                p_src, src_offset, p_dst, dst_offset);
        });

        static_if<DstInMemOp == InMemoryDataOperation::AtomicAdd>{}([&](auto) {
            AtomicAddData<T, DataPerAccess>{}.template Run<SrcAddressSpace, DstAddressSpace>(
                p_src, src_offset, p_dst, dst_offset);
        });
    }
    else
    {
        for(index_t i = 0; i < DataPerAccess; i++)
        {
            // TODO: use static_if::ElseIf
            static_if<DstInMemOp == InMemoryDataOperation::Set>{}([&](auto) {
                SetData<T, 1>{}.template Run<SrcAddressSpace, DstAddressSpace>(
                    p_src, src_offset + i * SrcDataStride, p_dst, dst_offset + i * DstDataStride);
            });

            static_if<DstInMemOp == InMemoryDataOperation::AtomicAdd>{}([&](auto) {
                AtomicAddData<T, 1>{}.template Run<SrcAddressSpace, DstAddressSpace>(
                    p_src, src_offset + i * SrcDataStride, p_dst, dst_offset + i * DstDataStride);
            });
        }
    }
}

} // namespace ck
#endif
