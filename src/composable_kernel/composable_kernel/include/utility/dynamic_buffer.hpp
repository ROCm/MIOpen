#ifndef CK_BUFFER_HPP
#define CK_BUFFER_HPP

#include "amd_buffer_addressing.hpp"
#include "c_style_pointer_cast.hpp"
#include "enable_if.hpp"

namespace ck {

template <AddressSpaceEnum_t BufferAddressSpace,
          typename T,
          typename ElementSpaceSize,
          bool InvalidElementUseNumericalZeroValue>
struct DynamicBuffer
{
    using type = T;

    T* p_data_;
    ElementSpaceSize element_space_size_;
    T invalid_element_value_ = T{0};

    __host__ __device__ constexpr DynamicBuffer(T* p_data, ElementSpaceSize element_space_size)
        : p_data_{p_data}, element_space_size_{element_space_size}
    {
    }

    __host__ __device__ constexpr DynamicBuffer(T* p_data,
                                                ElementSpaceSize element_space_size,
                                                T invalid_element_value)
        : p_data_{p_data},
          element_space_size_{element_space_size},
          invalid_element_value_{invalid_element_value}
    {
    }

    __host__ __device__ static constexpr AddressSpaceEnum_t GetAddressSpace()
    {
        return BufferAddressSpace;
    }

    __host__ __device__ constexpr T& operator[](index_t i) const { return p_data_[i]; }

    template <typename X,
              typename enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type>::value,
                  bool>::type = false>
    __host__ __device__ constexpr auto Get(index_t i, bool is_valid_element) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector =
            scalar_type<remove_cv_t<remove_reference_t<T>>>::vector_size;

        constexpr index_t scalar_per_x_vector =
            scalar_type<remove_cv_t<remove_reference_t<X>>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X need to be multiple T");

#if CK_USE_AMD_BUFFER_ADDRESSING
        bool constexpr use_amd_buffer_addressing = true;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(GetAddressSpace() == AddressSpaceEnum_t::Global && use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return amd_buffer_load_invalid_element_return_return_zero<
                    remove_cv_t<remove_reference_t<T>>,
                    t_per_x>(p_data_, i, is_valid_element, element_space_size_);
            }
            else
            {
                return amd_buffer_load_invalid_element_return_customized_value<
                    remove_cv_t<remove_reference_t<T>>,
                    t_per_x>(
                    p_data_, i, is_valid_element, element_space_size_, invalid_element_value_);
            }
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return is_valid_element ? *c_style_pointer_cast<const X*>(&p_data_[i]) : X{0};
            }
            else
            {
                return is_valid_element ? *c_style_pointer_cast<const X*>(&p_data_[i])
                                        : X{invalid_element_value_};
            }
        }
    }

    template <typename X,
              typename enable_if<
                  is_same<typename scalar_type<remove_cv_t<remove_reference_t<X>>>::type,
                          typename scalar_type<remove_cv_t<remove_reference_t<T>>>::type>::value,
                  bool>::type = false>
    __host__ __device__ void Set(index_t i, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector =
            scalar_type<remove_cv_t<remove_reference_t<T>>>::vector_size;

        constexpr index_t scalar_per_x_vector =
            scalar_type<remove_cv_t<remove_reference_t<X>>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X need to be multiple T");

        if constexpr(GetAddressSpace() == AddressSpaceEnum_t::Global)
        {
#if CK_USE_AMD_BUFFER_ADDRESSING
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_store<remove_cv_t<remove_reference_t<T>>, t_per_x>(
                x, p_data_, i, is_valid_element, element_space_size_);
#else
            if(is_valid_element)
            {
                *c_style_pointer_cast<X*>(&p_data_[i]) = x;
            }
#endif
        }
        else if constexpr(GetAddressSpace() == AddressSpaceEnum_t::Lds)
        {
            if(is_valid_element)
            {
#if !CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
                *c_style_pointer_cast<X*>(&p_data_[i]) = x;
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
                        *c_style_pointer_cast<int8_t*>(&p_data_[i]) =
                            *c_style_pointer_cast<const int8_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>, int8_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x2_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int16_t*>(&p_data_[i]) =
                            *c_style_pointer_cast<const int16_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>, int8_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x4_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                            *c_style_pointer_cast<const int32_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>,
                                              int8x4_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x4_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32_t*>(&p_data_[i]) =
                            *c_style_pointer_cast<const int32_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>,
                                              int8x8_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x8_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32x2_t*>(&p_data_[i]) =
                            *c_style_pointer_cast<const int32x2_t*>(&x);
                    }
                    else if constexpr(is_same<remove_cv_t<remove_reference_t<T>>,
                                              int8x16_t>::value &&
                                      is_same<remove_cv_t<remove_reference_t<X>>, int8x16_t>::value)
                    {
                        // HACK: cast pointer of x is bad
                        // TODO: remove this after compiler fix
                        *c_style_pointer_cast<int32x4_t*>(&p_data_[i]) =
                            *c_style_pointer_cast<const int32x4_t*>(&x);
                    }
                }
                else
                {
                    *c_style_pointer_cast<X*>(&p_data_[i]) = x;
                }
#endif
            }
        }
        else
        {
            if(is_valid_element)
            {
                *c_style_pointer_cast<X*>(&p_data_[i]) = x;
            }
        }
    }

    __host__ __device__ static constexpr bool IsStaticBuffer() { return false; }

    __host__ __device__ static constexpr bool IsDynamicBuffer() { return true; }
};

template <AddressSpaceEnum_t BufferAddressSpace, typename T, typename ElementSpaceSize>
__host__ __device__ constexpr auto make_dynamic_buffer(T* p, ElementSpaceSize element_space_size)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize, true>{p, element_space_size};
}

template <
    AddressSpaceEnum_t BufferAddressSpace,
    typename T,
    typename ElementSpaceSize,
    typename X,
    typename enable_if<is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value, bool>::type = false>
__host__ __device__ constexpr auto
make_dynamic_buffer(T* p, ElementSpaceSize element_space_size, X invalid_element_value)
{
    return DynamicBuffer<BufferAddressSpace, T, ElementSpaceSize, false>{
        p, element_space_size, invalid_element_value};
}

} // namespace ck
#endif
