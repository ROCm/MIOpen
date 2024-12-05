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

template <typename T, std::size_t N>
class InlineVector
{
public:
    using storage_type           = std::array<T, N>;
    using value_type             = typename storage_type::value_type;
    using size_type              = typename storage_type::size_type;
    using reference              = typename storage_type::reference;
    using const_reference        = typename storage_type::const_reference;
    using pointer                = typename storage_type::pointer;
    using const_pointer          = typename storage_type::const_pointer;
    using iterator               = typename storage_type::iterator;
    using const_iterator         = typename storage_type::const_iterator;
    using reverse_iterator       = typename storage_type::reverse_iterator;
    using const_reverse_iterator = typename storage_type::const_reverse_iterator;
    static_assert(std::is_scalar_v<T>, "InlineVector currently supports scalar type only");

    // Default constructor
    InlineVector() = default;

    // Copy and move constructor
    InlineVector(const InlineVector& inline_vec)     = default;
    InlineVector(InlineVector&& inline_vec) noexcept = default;

    InlineVector(std::initializer_list<T> data) : real_size(data.size())
    {
        if(real_size > N)
        {
            MIOPEN_THROW("Input data size is bigger than InlineVector's capacity");
        }

        std::copy(data.begin(), data.end(), storage.begin());
    }

    template <typename InputIterator>
    InlineVector(InputIterator first, InputIterator last) : real_size(std::distance(first, last))
    {
        if(real_size > N)
        {
            MIOPEN_THROW("Input data size is bigger than InlineVector's capacity");
        }

        std::copy(first, last, storage.begin());
    }

    // Copy/move operator
    InlineVector& operator=(const InlineVector& inline_vec) = default;
    InlineVector& operator=(InlineVector&& inline_vec) noexcept = default;

    // Iterators
    iterator begin() noexcept { return iterator(data()); }

    const_iterator begin() const noexcept { return const_iterator(data()); }

    iterator end() noexcept { return iterator(data() + real_size); }

    const_iterator end() const noexcept { return const_iterator(data() + real_size); }

    // Constant iterator
    const_iterator cbegin() const noexcept { return const_iterator(data()); }

    const_iterator cend() const noexcept { return const_iterator(data() + real_size); }

    // Reverse iterators
    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

    // Constant reverse iterators
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

    // Element access
    reference operator[](std::size_t n) noexcept
    {
        assert(n < real_size);
        return storage[n];
    }

    const_reference operator[](std::size_t n) const noexcept
    {
        assert(n < real_size);
        return storage[n];
    }

    // Element access with boundaries check
    reference at(std::size_t n)
    {
        if(n >= real_size)
        {
            MIOPEN_THROW("Access to InlineVector is out of range");
        }
        return storage.at(n);
    }

    const_reference at(std::size_t n) const
    {
        if(n >= real_size)
        {
            MIOPEN_THROW("Access to InlineVector is out of range");
        }
        return storage.at(n);
    }

    // Access to first element
    reference front()
    {
        if(empty())
        {
            MIOPEN_THROW("Cannot get front element, InlineVector is empty");
        }
        return (*begin());
    }

    const_reference front() const
    {
        if(empty())
        {
            MIOPEN_THROW("Cannot get front element, InlineVector is empty");
        }
        return (*begin());
    }

    // Access to last element
    reference back()
    {
        if(empty())
        {
            MIOPEN_THROW("Cannot get back element, InlineVector is empty");
        }
        return *std::prev(end());
    }

    const_reference back() const
    {
        if(empty())
        {
            MIOPEN_THROW("Cannot get back element, InlineVector is empty");
        }
        return *std::prev(end());
    }

    // Pointer to start of array
    pointer data() noexcept { return storage.data(); }

    const_pointer data() const noexcept { return storage.data(); }

    // Resize
    void resize(std::size_t n) { resize(n, T{}); }

    void resize(std::size_t n, const T& v)
    {
        if(n > N)
        {
            MIOPEN_THROW("It is not possible to resize beyond capacity");
        }

        if(n > real_size)
        {
            std::fill(begin() + real_size, begin() + n, v);
        }

        real_size = n;
    }

    // Add element to the back
    void push_back(const T& e)
    {
        if(real_size == N)
        {
            MIOPEN_THROW("InlineVector already full");
        }
        storage[real_size++] = e;
    }

    void push_back(const T&& e)
    {
        if(real_size == N)
        {
            MIOPEN_THROW("InlineVector already full");
        }
        storage[real_size++] = std::move(e);
    }

    /*
        Because only scalar type is supported there is no need for emplace_back method.
        Implement emplace_back method when adding support for other data types.
    */

    // Remove element from the back
    void pop_back() noexcept { real_size = (real_size > 1) ? (real_size - 1) : 0; }

    // Clear
    void clear() noexcept { real_size = 0; }

    // Empty
    bool empty() const noexcept { return real_size == 0; }

    // Real size
    size_type size() const noexcept { return real_size; }

    // Capacity
    constexpr size_type capacity() const { return N; }

private:
    storage_type storage{};
    size_type real_size = 0;
};

} // namespace miopen

#endif
