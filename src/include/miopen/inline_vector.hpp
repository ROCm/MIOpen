/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_INLINE_VECTOR_HPP
#define GUARD_MIOPEN_INLINE_VECTOR_HPP

#include <array>
#include <miopen/config.h>
#include <miopen/errors.hpp>

namespace miopen {

template <typename T, uint8_t N>
class InlineVector
{
public:
    using value_type = T;

    InlineVector() noexcept : real_size(0){};

    InlineVector(const InlineVector& inline_vec)     = default;
    InlineVector(InlineVector&& inline_vec) noexcept = default;

    InlineVector(std::initializer_list<T> _data) : real_size(_data.size())
    {
        if(_data.size() > N)
        {
            MIOPEN_THROW("Input data size is bigger than InlineVector's capacity");
        }
        std::copy(_data.begin(), _data.end(), data.begin());
    }

    template <typename _InputIterator, typename = std::_RequireInputIter<_InputIterator>>
    InlineVector(_InputIterator first, _InputIterator last)
    {
        if(std::distance(first, last) > N)
        {
            MIOPEN_THROW("Input data size is bigger than InlineVector's capacity");
        }
        std::copy(first, last, data.begin());
        real_size = std::distance(first, last);
    }

    // Iterators
    T* begin() { return data.begin(); }

    T* end() { return (data.begin() + real_size); }

    // Reverse iterators
    std::reverse_iterator<T*> rbegin() { return std::reverse_iterator<T*>(end()); }

    std::reverse_iterator<T*> rend() { return std::reverse_iterator<T*>(begin()); }

    // Element access
    T& operator[](std::size_t n) { return data[n]; }

    const T& operator[](std::size_t n) const { return data[n]; }

    // Empty
    bool empty() const { return real_size == 0; }

    // Real size
    uint8_t size() const { return real_size; }

    // Capacity
    constexpr uint8_t capacity() const { return N; }

private:
    std::array<T, N> data;
    uint8_t real_size;
};

} // namespace miopen

#endif
