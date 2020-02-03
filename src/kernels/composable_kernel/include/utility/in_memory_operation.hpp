#ifndef CK_IN_MEMORY_OPERATION_AMD_HPP
#define CK_IN_MEMORY_OPERATION_AMD_HPP

#include "float_type.hpp"
#include "amd_buffer_addressing.hpp"

namespace ck {

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
                *reinterpret_cast<const vector_t*>(&p_src[src_offset]), p_dst, dst_offset, 0);
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
    using vector_t = typename vector_type<T, DataPerAccess>::MemoryType;

    static_if<SrcAddressSpace == AddressSpace::Vgpr &&
              DstAddressSpace == AddressSpace::Global>{}([&](auto) {
#if CK_USE_AMD_BUFFER_ATOMIC_ADD
        amd_intrinsic_buffer_atomic_add<T, DataPerAccess>(
            *reinterpret_cast<const vector_t*>(&p_src[src_offset]), p_dst, dst_offset, 0);
#else
        atomicAdd(reinterpret_cast<vector_t*>(&p_dst[dst_offset]),
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
          InMemoryDataOperation DstInMemOp>
__device__ void transfer_data(const T* p_src, index_t src_offset, T* p_dst, index_t dst_offset)
{
    static_assert(DstInMemOp == InMemoryDataOperation::Set ||
                      DstInMemOp == InMemoryDataOperation::AtomicAdd,
                  "wrong! InMemoryDataOperation not supported!");

    // TODO: use static_if::ElseIf
    static_if<DstInMemOp == InMemoryDataOperation::Set>{}([&](auto) {
        set_data<T, DataPerAccess, SrcAddressSpace, DstAddressSpace>(
            p_src, src_offset, p_dst, dst_offset);
    });

    static_if<DstInMemOp == InMemoryDataOperation::AtomicAdd>{}([&](auto) {
        atomic_add_data<T, DataPerAccess, SrcAddressSpace, DstAddressSpace>(
            p_src, src_offset, p_dst, dst_offset);
    });
}

} // namespace ck
#endif
