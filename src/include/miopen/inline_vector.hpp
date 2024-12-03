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
    using value_type             = storage_type::value_type;
    using size_type              = storage_type::size_type;
    using reference              = storage_type::reference;
    using const_reference        = storage_type::const_reference;
    using pointer                = storage_type::pointer;
    using const_pointer          = storage_type::const_pointer;
    using iterator               = storage_type::iterator;
    using const_iterator         = storage_type::const_iterator;
    using reverse_iterator       = storage_type::reverse_iterator;
    using const_reverse_iterator = storage_type::const_reverse_iterator;
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

    template <typename _InputIterator>
    InlineVector(_InputIterator first, _InputIterator last) : real_size(std::distance(first, last))
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
    iterator begin() noexcept { return storage.begin(); }

    const_iterator begin() const noexcept { return storage.begin(); }

    iterator end() noexcept { return (storage.begin() + real_size); }

    const_iterator end() const noexcept { return (storage.begin() + real_size); }

    // Constant iterator
    const_iterator cbegin() const noexcept { return begin(); }

    const_iterator cend() const noexcept { return end(); }

    // Reverse iterators
    reverse_iterator rbegin() noexcept { return std::reverse_iterator<T*>(end()); }

    const_reverse_iterator rbegin() const noexcept
    {
        return std::reverse_iterator<const T*>(end());
    }

    reverse_iterator rend() noexcept { return std::reverse_iterator<T*>(begin()); }

    const_reverse_iterator rend() const noexcept
    {
        return std::reverse_iterator<const T*>(begin());
    }

    // Constant reverse iterators
    const_reverse_iterator crbegin() const noexcept
    {
        return std::reverse_iterator<const T*>(cend());
    }

    const_reverse_iterator crend() const noexcept
    {
        return std::reverse_iterator<const T*>(cbegin());
    }

    // Element access
    reference operator[](std::size_t n) noexcept { return storage[n]; }

    const_reference operator[](std::size_t n) const noexcept { return storage[n]; }

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

    // Create element and add it to the back
    template <typename... _Args>
    void emplace_back(_Args&&... args)
    {
        if(real_size == N)
        {
            MIOPEN_THROW("InlineVector already full");
        }
        storage[real_size++] = T(std::forward<_Args>(args)...);
    }

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
