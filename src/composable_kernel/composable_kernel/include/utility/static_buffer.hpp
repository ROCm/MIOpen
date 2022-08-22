/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
