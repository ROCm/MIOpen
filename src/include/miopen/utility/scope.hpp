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

#pragma once

#include <type_traits>
#include <utility>

namespace miopen {

/* The implementation of scope_exit https://en.cppreference.com/w/cpp/experimental/scope_exit
 * Once it is in STL we can remove this implementation
 */

template <typename ExitHandler>
class scope_exit
{
    ExitHandler mHandler;
    bool mActive = true;

public:
    explicit scope_exit(const ExitHandler& eh) noexcept : mHandler(eh)
    {
        static_assert(noexcept(ExitHandler(eh)), "A Handler with throwing ctor is useless");
    }
    explicit scope_exit(ExitHandler&& eh) noexcept : mHandler(std::move(eh))
    {
        static_assert(noexcept(ExitHandler(std::move(eh))),
                      "A Handler with throwing ctor is useless");
    }

    ~scope_exit()
    {
        if(mActive)
        {
            mHandler();
        }
    }

    scope_exit(const scope_exit&) = delete;
    scope_exit& operator=(const scope_exit&) = delete;
    void release() { mActive = false; };
};

template <typename ExitHandler>
scope_exit(const ExitHandler&) -> scope_exit<std::decay_t<ExitHandler>>;

template <typename ExitHandler>
scope_exit(ExitHandler&&) -> scope_exit<std::decay_t<ExitHandler>>;

} // namespace miopen
