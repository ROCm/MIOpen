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
    using value_type = T;
    static_assert(std::is_scalar_v<T>, "Input data size is bigger than InlineVector's capacity");

    // Default constructor
    InlineVector() = default;

    // Copy and move constructor
    InlineVector(const InlineVector& inline_vec)     = default;
    InlineVector(InlineVector&& inline_vec) noexcept = default;

    InlineVector(std::initializer_list<T> __data) : real_size(__data.size())
    {
        if(real_size > N)
        {
            MIOPEN_THROW("Input data size is bigger than InlineVector's capacity");
        }

        std::copy(__data.begin(), __data.end(), _data.begin());
    }

    template <typename _InputIterator, typename = std::_RequireInputIter<_InputIterator>>
    InlineVector(_InputIterator first, _InputIterator last) : real_size(std::distance(first, last))
    {
        if(real_size > N)
        {
            MIOPEN_THROW("Input data size is bigger than InlineVector's capacity");
        }

        std::copy(first, last, _data.begin());
    }

    // Copy/move operator
    InlineVector& operator=(const InlineVector& inline_vec)     = default;
    InlineVector& operator=(InlineVector&& inline_vec) noexcept = default;

    // Iterators
    T* begin() noexcept { return _data.begin(); }

    const T* begin() const noexcept { return _data.begin(); }

    T* end() noexcept { return (_data.begin() + real_size); }

    const T* end() const noexcept { return (_data.begin() + real_size); }

    // Constant iterator
    const T* cbegin() const noexcept { return begin(); }

    const T* cend() const noexcept { return end(); }

    // Reverse iterators
    std::reverse_iterator<T*> rbegin() noexcept { return std::reverse_iterator<T*>(end()); }

    std::reverse_iterator<const T*> rbegin() const noexcept
    {
        return std::reverse_iterator<const T*>(end());
    }

    std::reverse_iterator<T*> rend() noexcept { return std::reverse_iterator<T*>(begin()); }

    std::reverse_iterator<const T*> rend() const noexcept
    {
        return std::reverse_iterator<const T*>(begin());
    }

    // Constant reverse iterators
    std::reverse_iterator<const T*> crbegin() const noexcept
    {
        return std::reverse_iterator<const T*>(cend());
    }

    std::reverse_iterator<const T*> crend() const noexcept
    {
        return std::reverse_iterator<const T*>(cbegin());
    }

    // Element access
    T& operator[](std::size_t n) noexcept { return _data[n]; }

    const T& operator[](std::size_t n) const noexcept { return _data[n]; }

    // Element access with boundaries check
    T& at(std::size_t n)
    {
        if(n >= real_size)
        {
            MIOPEN_THROW("Access to InlineVector is out of range");
        }
        return _data.at(n);
    }

    const T& at(std::size_t n) const
    {
        if(n >= real_size)
        {
            MIOPEN_THROW("Access to InlineVector is out of range");
        }
        return _data.at(n);
    }

    // Access to first element
    T& front()
    {
        if(empty())
        {
            MIOPEN_THROW("Cannot get front element, InlineVector is empty");
        }
        return (*begin());
    }

    const T& front() const
    {
        if(empty())
        {
            MIOPEN_THROW("Cannot get front element, InlineVector is empty");
        }
        return (*begin());
    }

    // Access to last element
    T& back()
    {
        if(empty())
        {
            MIOPEN_THROW("Cannot get back element, InlineVector is empty");
        }
        return *std::prev(end());
    }

    const T& back() const
    {
        if(empty())
        {
            MIOPEN_THROW("Cannot get back element, InlineVector is empty");
        }
        return *std::prev(end());
    }

    // Pointer to start of array
    T* data() noexcept { return _data.data(); }

    const T* data() const noexcept { return _data.data(); }

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
        _data[real_size++] = e;
    }

    void push_back(const T&& e)
    {
        if(real_size == N)
        {
            MIOPEN_THROW("InlineVector already full");
        }
        _data[real_size++] = std::move(e);
    }

    // Create element and add it to the back
    template <typename... _Args>
    void emplace_back(_Args&&... args)
    {
        if(real_size == N)
        {
            MIOPEN_THROW("InlineVector already full");
        }
        _data[real_size++] = T(std::forward<_Args>(args)...);
    }

    // Remove element from the back
    void pop_back() noexcept { real_size = (real_size > 1) ? (real_size - 1) : 0; }

    // Clear
    void clear() noexcept { real_size = 0; }

    // Empty
    bool empty() const noexcept { return real_size == 0; }

    // Real size
    std::size_t size() const noexcept { return real_size; }

    // Capacity
    constexpr std::size_t capacity() const { return N; }

private:
    std::array<T, N> _data{};
    std::size_t real_size = 0;
};

} // namespace miopen

#endif
