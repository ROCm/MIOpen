#ifndef CK_DYNAMIC_BUFFER_HPP
#define CK_DYNAMIC_BUFFER_HPP

namespace ck {

#include "amd_buffer_addressing_v2.hpp"

template <AddressSpaceEnum_t BufferAddressSpace, typename T, typename ElementSpaceSize>
struct DynamicBuffer
{
    using type = T;

    T* p_data_;
    ElementSpaceSize element_space_size_;

    __host__ __device__ constexpr DynamicBuffer(T* p_data, ElementSpaceSize element_space_size)
        : p_data_{p_data}, element_space_size_{element_space_size}
    {
    }

    __host__ __device__ static constexpr AddressSpaceEnum_t GetAddressSpace()
    {
        return BufferAddressSpace;
    }

    __host__ __device__ constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    __host__ __device__ constexpr T& operator()(index_t i) { return p_data_[i]; }

    template <typename X,
              typename std::enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type>::value,
                  bool>::type = false>
    __host__ __device__ constexpr auto Get(index_t i, bool is_valid_offset) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector =
            scalar_type<remove_cv_t<remove_reference_t<T>>>::vector_size;

        constexpr index_t scalar_per_x_vector =
            scalar_type<remove_cv_t<remove_reference_t<X>>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X need to be multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        if constexpr(GetAddressSpace() == AddressSpaceEnum_t::Global)
        {
#if CK_USE_AMD_BUFFER_ADDRESSING
            return amd_buffer_load_v2<remove_cv_t<remove_reference_t<T>>, t_per_x>(
                p_data_, i, is_valid_offset, element_space_size_);
#else
            return is_valid_offset ? *reinterpret_cast<const X*>(&p_data_[i]) : X{0};
#endif
        }
        else
        {
            return is_valid_offset ? *reinterpret_cast<const X*>(&p_data_[i]) : X{0};
        }
    }

    template <typename X,
              typename std::enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type>::value,
                  bool>::type = false>
    __host__ __device__ void Set(index_t i, bool is_valid_offset, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector =
            scalar_type<remove_cv_t<remove_reference_t<T>>>::vector_size;

        constexpr index_t scalar_per_x_vector =
            scalar_type<remove_cv_t<remove_reference_t<X>>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X need to be multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        if constexpr(GetAddressSpace() == AddressSpaceEnum_t::Global)
        {
#if CK_USE_AMD_BUFFER_ADDRESSING
            amd_buffer_store_v2<remove_cv_t<remove_reference_t<T>>, t_per_x>(
                x, p_data_, i, is_valid_offset, element_space_size_);
#else
            if(is_valid_offset)
            {
                *reinterpret_cast<X*>(&p_data_[i]) = x;
            }
#endif
        }
        else if constexpr(GetAddressSpace() == AddressSpaceEnum_t::Lds)
        {
            if(is_valid_offset)
            {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
                *reinterpret_cast<X*>(&p_data_[i]) = x;
#else
                // HACK: compiler would lower IR "store<i8, 16> address_space(3)" into
                // inefficient
                // ISA, so I try to let compiler emit IR "store<i32, 4>" which would be lower to
                // ds_write_b128
                // TODO: remove this after compiler fix
                if constexpr(is_same<typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type,
                                     int8_t>::value)
                {
                    static_assert(
                        (is_same<remove_cv_t<remove_reference_t<T>>, int8_t>::value &&
                         is_same<remove_cv_t<remove_reference_t<X>>, int8_t>::value) ||
                            (is_same<remove_cv_t<remove_reference_t<T>>, int8_t>::value &&
                             is_same<remove_cv_t<remove_reference_t<X>>, int8x2_t>::value) ||
                            (is_same<remove_cv_t<remove_reference_t<T>>, int8_t>::value &&
                             is_same<remove_cv_t<remove_reference_t<X>>, int8x4_t>::value) ||
                            (is_same<remove_cv_t<remove_reference_t<T>>, int8x4_t>::value &&
                             is_same<remove_cv_t<remove_reference_t<X>>, int8x4_t>::value) ||
                            (is_same<remove_cv_t<remove_reference_t<T>>, int8x8_t>::value &&
                             is_same<remove_cv_t<remove_reference_t<X>>, int8x8_t>::value) ||
                            (is_same<remove_cv_t<remove_reference_t<T>>, int8x16_t>::value &&
                             is_same<remove_cv_t<remove_reference_t<X>>, int8x16_t>::value),
                        "wrong! not implemented for this combination, please add "
                        "implementation");

                    if constexpr(is_same<remove_cv_t<remove_reference_t<T>>, int8_t>::value &&
                                 is_same<remove_cv_t<remove_reference_t<X>>, int8_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *reinterpret_cast<int8_t*>(&p_data_[i]) =
                            *reinterpret_cast<const int8_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>, int8_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x2_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *reinterpret_cast<int16_t*>(&p_data_[i]) =
                            *reinterpret_cast<const int16_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>, int8_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x4_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *reinterpret_cast<int32_t*>(&p_data_[i]) =
                            *reinterpret_cast<const int32_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>,
                                              int8x4_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x4_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *reinterpret_cast<int32_t*>(&p_data_[i]) =
                            *reinterpret_cast<const int32_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>,
                                              int8x8_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x8_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *reinterpret_cast<int32x2_t*>(&p_data_[i]) =
                            *reinterpret_cast<const int32x2_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>,
                                              int8x16_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x16_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *reinterpret_cast<int32x4_t*>(&p_data_[i]) =
                            *reinterpret_cast<const int32x4_t*>(&x);
                    }
                }
                else
                {
                    *reinterpret_cast<X*>(&p_data_[i]) = x;
                }
#endif
            }
        }
        else
        {
            if(is_valid_offset)
            {
                *reinterpret_cast<X*>(&p_data_[i]) = x;
            }
        }
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return false; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return true; }
};

template <AddressSpaceEnum_t BufferAddressSpace = AddressSpaceEnum_t::Generic,
          typename T,
          typename ElementSpaceSize>
__host__ __device__ constexpr auto make_dynamic_buffer(T* p, ElementSpaceSize element_space_size)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize>{p, element_space_size};
}

} // namespace ck
#endif
