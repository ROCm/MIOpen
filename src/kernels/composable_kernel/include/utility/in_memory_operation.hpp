#ifndef CK_IN_MEMORY_OPERATION_AMD_HPP
#define CK_IN_MEMORY_OPERATION_AMD_HPP

#include "float_type.hpp"
#include "amd_buffer_addressing.hpp"

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

template <typename T,
          index_t DataPerAccess,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace>
__device__ void set_data(const T* p_src, index_t src_offset, T* p_dst, index_t dst_offset)
{
    using vector_t = typename vector_type<T, DataPerAccess>::MemoryType;

#if CK_USE_AMD_BUFFER_ADDRESSING
    // TODO: use static_if::ElseIf, instead of nested static_if
    static_if<SrcAddressSpace == AddressSpace::Global &&
              DstAddressSpace == AddressSpace::Vgpr>{}([&](auto) {
        // buffer_load requires:
        //   1) p_src must be in global memory space, d_dst must be vgpr
        //   2) p_src to be a block-invariant pointer.
        // It is user's responsibility to make sure that is true.
        *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) =
            amd_intrinsic_buffer_load<T, DataPerAccess>(p_src, src_offset, 0);
    }).Else([&](auto) {
        static_if<SrcAddressSpace == AddressSpace::Vgpr &&
                  DstAddressSpace == AddressSpace::Global>{}([&](auto) {
            // buffer_store requires:
            //   1) p_src must be in vgpr space, d_dst must be global memory
            //   2) p_dst to be a block-invariant pointer.
            // It is user's responsibility to make sure that is true.
            amd_intrinsic_buffer_store<T, DataPerAccess>(
                &(p_src[src_offset]), p_dst, dst_offset, 0);
        }).Else([&](auto) {
            *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) =
                *reinterpret_cast<const vector_t*>(&p_src[src_offset]);
        });
    });
#else
    *reinterpret_cast<vector_t*>(&p_dst[dst_offset]) =
        *reinterpret_cast<const vector_t*>(&p_src[src_offset]);
#endif
}

template <typename T,
          index_t DataPerAccess,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace>
__device__ void atomic_add_data(const T* p_src, index_t src_offset, T* p_dst, index_t dst_offset)
{
    static_if<SrcAddressSpace == AddressSpace::Vgpr &&
              DstAddressSpace == AddressSpace::Global>{}([&](auto) {
#if CK_USE_AMD_BUFFER_ATOMIC_ADD
        amd_intrinsic_buffer_atomic_add<T, DataPerAccess>(
            &(p_src[src_offset]), p_dst, dst_offset, 0);
#else
        using vector_t = typename vector_type<T, DataPerAccess>::MemoryType;

        atomic_add_impl(reinterpret_cast<vector_t*>(&p_dst[dst_offset]),
                        *reinterpret_cast<const vector_t*>(&p_src[src_offset]));
#endif
    }).Else([&](auto fwd) {
        static_assert(fwd(false), "atomic_add doesn't support this memory space");
    });
}

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

    static_if<(SrcDataStride > 1 || DstDataStride > 1)>{}([&](auto) {

        for(index_t j = 0; j < DataPerAccess; j++)
        {
            // TODO: use static_if::ElseIf
            static_if<DstInMemOp == InMemoryDataOperation::Set>{}([&](auto) {
                set_data<T, 1, SrcAddressSpace, DstAddressSpace>(
                    p_src, src_offset + j * SrcDataStride, p_dst, dst_offset + j * DstDataStride);
            });

            static_if<DstInMemOp == InMemoryDataOperation::AtomicAdd>{}([&](auto) {
                atomic_add_data<T, 1, SrcAddressSpace, DstAddressSpace>(
                    p_src, src_offset + j * SrcDataStride, p_dst, dst_offset + j * DstDataStride);
            });
        }

    }).Else([&](auto) {

        // TODO: use static_if::ElseIf
        static_if<DstInMemOp == InMemoryDataOperation::Set>{}([&](auto) {
            set_data<T, DataPerAccess, SrcAddressSpace, DstAddressSpace>(
                p_src, src_offset, p_dst, dst_offset);
        });

        static_if<DstInMemOp == InMemoryDataOperation::AtomicAdd>{}([&](auto) {
            atomic_add_data<T, DataPerAccess, SrcAddressSpace, DstAddressSpace>(
                p_src, src_offset, p_dst, dst_offset);
        });

    });
}

} // namespace ck
#endif
