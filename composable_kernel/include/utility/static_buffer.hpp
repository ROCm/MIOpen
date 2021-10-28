#ifndef CK_STATIC_BUFFER_HPP
#define CK_STATIC_BUFFER_HPP

#include "statically_indexed_array.hpp"

namespace ck {

template <AddressSpaceEnum_t BufferAddressSpace,
          typename T,
          index_t N,
          bool InvalidElementUseNumericalZeroValue>
struct StaticBuffer : public StaticallyIndexedArray<T, N>
{
    using type = T;
    using base = StaticallyIndexedArray<T, N>;

    T invalid_element_value_ = T{0};

    __host__ __device__ constexpr StaticBuffer() : base{} {}

    __host__ __device__ constexpr StaticBuffer(T invalid_element_value)
        : base{}, invalid_element_value_{invalid_element_value}
    {
    }

    __host__ __device__ static constexpr AddressSpaceEnum_t GetAddressSpace()
    {
        return BufferAddressSpace;
    }

    template <index_t I>
    __host__ __device__ constexpr auto Get(Number<I> i, bool is_valid_element) const
    {
        if constexpr(InvalidElementUseNumericalZeroValue)
        {
            return is_valid_element ? At(i) : T{0};
        }
        else
        {
            return is_valid_element ? At(i) : invalid_element_value_;
        }
    }

    template <index_t I>
    __host__ __device__ void Set(Number<I> i, bool is_valid_element, const T& x)
    {
        if(is_valid_element)
        {
            At(i) = x;
        }
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }
};

template <AddressSpaceEnum_t BufferAddressSpace,
          typename T,
          index_t N,
          bool InvalidElementUseNumericalZeroValue>
struct StaticBufferV2 : public StaticallyIndexedArray<T, N>
{
    using type = T;
    using base = StaticallyIndexedArray<T, N>;

    using VecBaseType = typename T::d1_t;

    __host__ __device__ static constexpr index_t GetVectorSize()
    {
        return sizeof(typename T::type) / sizeof(VecBaseType);
    }

    static constexpr index_t vector_size = GetVectorSize();

    VecBaseType invalid_element_value_ = VecBaseType{0};

    T invalid_vec_value_ = T{0};

    __host__ __device__ constexpr StaticBufferV2() : base{} {}

    __host__ __device__ constexpr StaticBufferV2(VecBaseType invalid_element_value)
        : base{},
          invalid_vec_value_{invalid_element_value},
          invalid_element_value_{invalid_element_value}
    {
    }

    __host__ __device__ static constexpr AddressSpaceEnum_t GetAddressSpace()
    {
        return BufferAddressSpace;
    }

    template <index_t I>
    __host__ __device__ constexpr auto& GetVector(Number<I> vec_id)
    {
        return this->At(vec_id);
    }

    template <index_t I>
    __host__ __device__ constexpr const auto& GetVector(Number<I> vec_id) const
    {
        return this->At(vec_id);
    }

    template <index_t I>
    __host__ __device__ constexpr auto& GetElement(Number<I> i, bool)
    {
        constexpr auto vec_id  = Number<i / vector_size>{};
        constexpr auto vec_off = Number<i % vector_size>{};

        return this->At(vec_id).template AsType<VecBaseType>()(vec_off);
    }

    template <index_t I>
    __host__ __device__ constexpr auto GetElement(Number<I> i, bool is_valid_element) const
    {
        constexpr auto vec_id  = Number<i / vector_size>{};
        constexpr auto vec_off = Number<i % vector_size>{};

        if constexpr(InvalidElementUseNumericalZeroValue)
        {
            return is_valid_element ? this->At(vec_id).template AsType<VecBaseType>()[vec_off]
                                    : VecBaseType{0};
        }
        else
        {
            return is_valid_element ? this->At(vec_id).template AsType<VecBaseType>()[vec_off]
                                    : invalid_element_value_;
        }
    }

    template <index_t I>
    __host__ __device__ constexpr auto operator[](Number<I> i) const
    {
        return GetElement(i, true);
    }

    template <index_t I>
    __host__ __device__ constexpr auto& operator()(Number<I> i)
    {
        return GetElement(i, true);
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return true; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return false; }
};

template <AddressSpaceEnum_t BufferAddressSpace, typename T, index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>)
{
    return StaticBuffer<BufferAddressSpace, T, N, true>{};
}

template <AddressSpaceEnum_t BufferAddressSpace, typename T, index_t N>
__host__ __device__ constexpr auto make_static_buffer(Number<N>, T invalid_element_value)
{
    return StaticBuffer<BufferAddressSpace, T, N, false>{invalid_element_value};
}

} // namespace ck
#endif
