#ifndef CK_STATIC_BUFFER_HPP
#define CK_STATIC_BUFFER_HPP

#include "statically_indexed_array.hpp"

namespace ck {

template <AddressSpaceEnum_t BufferAddressSpace, typename T, index_t N>
struct StaticBuffer : public StaticallyIndexedArray<T, N>
{
    using type = T;
    using base = StaticallyIndexedArray<T, N>;

    __host__ __device__ constexpr StaticBuffer() : base{} {}

    __host__ __device__ static constexpr AddressSpaceEnum_t GetAddressSpace()
    {
        return BufferAddressSpace;
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }
};

template <AddressSpaceEnum_t BufferAddressSpace = AddressSpaceEnum_t::Generic,
          typename T,
          index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>)
{
    return StaticBuffer<BufferAddressSpace, T, N>{};
}

} // namespace ck
#endif
